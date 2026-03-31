import os
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COVER_DIR = "data/datasets/cover"  # Directory for clean images
STEGO_DIR = "data/datasets/stego"  # Directory for steganographic images
RESULTS_DIR = "results"   # Directory for storing results

os.makedirs(RESULTS_DIR, exist_ok=True)
class StegDataset(Dataset):
    def __init__(self, cover_dir, stego_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load cover (clean) images
        if os.path.exists(cover_dir):
            cover_files = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.samples.extend(cover_files)
            self.labels.extend([0] * len(cover_files))  # 0 = cover/clean
        
        # Load stego images
        if os.path.exists(stego_dir):
            stego_files = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.samples.extend(stego_files)
            self.labels.extend([1] * len(stego_files))  # 1 = stego
        
        logger.info(f"Dataset loaded: {len(self.samples)} samples "
                   f"({self.labels.count(0)} cover, {self.labels.count(1)} stego)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply DCT transform to enhance steganographic features detection
            if self.transform:
                image = self.transform(image)
            
            return image, label, image_path
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a placeholder if image loading fails
            return torch.zeros((3, 224, 224)), label, image_path

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on relevant image regions"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Max pooling across channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension
        attention = torch.cat([avg_pool, max_pool], dim=1)
        # Apply convolution and sigmoid activation
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        # Apply attention to input tensor
        return x * attention

class CNNAnalyser:
    
    def __init__(self):
        self.image_size = (224, 224)  # Standard input size for most CNN models
        self.batch_size = 32
        self.detection_threshold = 0.5
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        self.cover_dir = COVER_DIR
        self.stego_dir = STEGO_DIR
        self.detection_result = {
            'is_stego': False,
            'confidence': 0.0,
            'features_analysis': {},
            'detection_time': 0.0,
            'file_path': '',
            'file_size': 0,
            'analysis_timestamp': '',
            'prediction_details': {}
        }
        
        # Regular transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Augmented transforms for training
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            # Minimal color jitter to preserve steganographic features
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        self.model = self._initialize_model()
        logger.info(f"CNN Analyser initialized with device: {self.device}")

    def _initialize_model(self):
        """Initialize a CNN model optimized for steganography detection"""
        try:
            # Start with a pre-trained EfficientNet model
            model = models.efficientnet_b0(weights="DEFAULT")
            
            # Replace classifier for binary classification
            num_ftrs = model.classifier[1].in_features
            
            # Add high-pass filter to enhance detection of pixel-level modifications
            with torch.no_grad():
                first_conv = list(model.features)[0][0]
                # Initialize some filters with high-pass kernels (SRM filters)
                srm_kernel = torch.tensor([
                    [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],  # Edge detection
                    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],   # Laplacian
                ], dtype=torch.float32).unsqueeze(1)
                
                if srm_kernel.shape[-1] == first_conv.kernel_size[0]:
                    srm_kernel = srm_kernel.repeat(3, 1, 1, 1)
                    first_conv.weight.data[:2] = srm_kernel
            
            # Add spatial attention module after features extractor
            spatial_attention = SpatialAttention()
            
            # Custom classifier with attention mechanism
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)  # 2 classes: cover(0), stego(1)
            )
            
            class StegDetectionModel(nn.Module):
                def __init__(self, backbone, attention, classifier):
                    super(StegDetectionModel, self).__init__()
                    self.features = backbone
                    self.attention = attention
                    self.classifier = classifier
                
                def forward(self, x):
                    x = self.features(x)
                    x = self.attention(x)
                    x = self.classifier(x)
                    return x
            
            model = StegDetectionModel(model.features, spatial_attention, classifier)
            model = model.to(self.device)
            model.eval()
            
            logger.info("Successfully initialized EfficientNet model with spatial attention")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing primary model: {e}")
            return self._create_backup_model()
    
    def _create_backup_model(self):
        """Create a simple backup model in case the main model fails to initialize"""
        logger.info("Creating backup CNN model")
        model = nn.Sequential(
            # First convolutional block with SRM-inspired filters
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Flatten
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 classes: cover(0), stego(1)
        )
        
        # Initialize SRM filters in first layer for better stego detection
        with torch.no_grad():
            srm_kernel = torch.tensor([
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],  # Edge detection filter
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],      # Laplacian filter
            ], dtype=torch.float32).view(2, 1, 3, 3)
            
            srm_kernel = srm_kernel.repeat(1, 3, 1, 1)
            model[0].weight.data[:2] = srm_kernel
        
        model.eval()
        model = model.to(self.device)
        logger.info("Backup CNN model created successfully")
        return model
    
    def train_model(self, epochs=20, learning_rate=0.001, save_path=None):
        """Train the model on steganography dataset"""
        logger.info("Starting model training...")
        
        # Check if directories exist
        if not os.path.exists(self.cover_dir) or not os.path.exists(self.stego_dir):
            logger.error("Training directories not found")
            return False
        
        # Create dataset and dataloader
        dataset = StegDataset(self.cover_dir, self.stego_dir, transform=self.train_transform)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model for training
        self.model.train()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for images, labels, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            epoch_val_loss = val_running_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            val_acc = correct / total
            val_accuracies.append(val_acc)
            
            # Update learning rate based on validation loss
            scheduler.step(epoch_val_loss)
            
            logger.info(f'Epoch [{epoch+1}/{epochs}], '
                       f'Train Loss: {epoch_train_loss:.4f}, '
                       f'Val Loss: {epoch_val_loss:.4f}, '
                       f'Val Accuracy: {val_acc*100:.2f}%')
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                if save_path is None:
                    save_path = os.path.join(RESULTS_DIR, 'best_stego_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_accuracy': val_acc
                }, save_path)
                logger.info(f"Model saved to {save_path}")
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')
        
        plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'))
        plt.close()
        
        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Training completed. Best validation accuracy: {checkpoint['val_accuracy']*100:.2f}%")
        return True
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess an image for prediction"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(self.device)
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def extract_features(self, img_tensor):
        """Extract features from image using the model's feature extractor"""
        try:
            with torch.no_grad():
                if hasattr(self.model, 'features'):
                    features = self.model.features(img_tensor)
                else:
                    # For sequential model, get features before classifier
                    temp_model = nn.Sequential(*list(self.model.children())[:-4])
                    features = temp_model(img_tensor)
                
                # Convert to numpy for analysis
                if features.dim() > 2:
                    features = features.mean([2, 3])  # Global average pooling
                
                return features.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.random.rand(1, 256)  # Return random features as fallback
    
    def extract_noise_residuals(self, image_path):
        """Extract noise residuals from image for steganalysis"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return {}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising to get estimate of the original image
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Calculate residual noise (potentially contains hidden data)
            residual = gray - denoised
            
            # Calculate statistical features from noise residual
            mean = np.mean(residual)
            std = np.std(residual)
            
            # Normalize residual for further analysis
            norm_residual = residual.astype('float32')
            if std > 0:
                norm_residual = (norm_residual - mean) / std
                skewness = np.mean(norm_residual ** 3)
                kurtosis = np.mean(norm_residual ** 4) - 3  # Excess kurtosis
            else:
                skewness = 0
                kurtosis = 0
            
            # Calculate histogram and entropy
            hist = cv2.calcHist([residual], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            non_zero = hist > 0
            entropy = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
            
            # DCT transform of residual to detect frequency domain anomalies
            dct = cv2.dct(np.float32(residual))
            dct_mean = np.mean(np.abs(dct))
            dct_std = np.std(np.abs(dct))
            
            # Return computed features
            return {
                'mean': float(mean),
                'std': float(std),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'entropy': float(entropy),
                'dct_mean': float(dct_mean),
                'dct_std': float(dct_std),
                'histogram_peaks': int(np.sum(hist > np.mean(hist) + np.std(hist))),
                'max_bin': int(np.argmax(hist))
            }
            
        except Exception as e:
            logger.error(f"Error extracting noise residuals: {e}")
            return {}
    
    def predict_image(self, image_path):
        """Predict if an image contains steganographic content"""
        logger.info(f"Analyzing image: {image_path}")
        start_time = time.time()
        
        self.detection_result = {
            'is_stego': False,
            'confidence': 0.0,
            'features_analysis': {},
            'detection_time': 0.0,
            'file_path': image_path,
            'file_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'prediction_details': {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Extract noise residuals
            noise_features = self.extract_noise_residuals(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
                is_stego = prediction.item() == 1
                confidence_value = confidence.item()
            
            # Extract CNN features for analysis
            features = self.extract_features(img_tensor)
            
            # Update detection result
            self.detection_result.update({
                'is_stego': is_stego,
                'confidence': confidence_value,
                'features_analysis': {
                    'cnn_features': features.tolist() if isinstance(features, np.ndarray) else features,
                    'noise_analysis': noise_features
                },
                'detection_time': time.time() - start_time,
                'prediction_details': {
                    'raw_output': outputs.cpu().numpy().tolist(),
                    'probabilities': probabilities.cpu().numpy().tolist(),
                    'prediction_class': prediction.item()
                }
            })
            
            logger.info(f"Analysis complete: {'STEGO' if is_stego else 'CLEAN'} with {confidence_value:.4f} confidence")
            return self.detection_result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.detection_result.update({
                'error': str(e),
                'detection_time': time.time() - start_time
            })
            return self.detection_result
    
    def visualize_features(self, image_path, output_path=None):
        """Create visualization of model's attention on the image"""
        try:
            if not output_path:
                output_path = os.path.join(RESULTS_DIR, f"vis_{Path(image_path).name}")
            
            # Load and preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Get original image for display
            orig_img = Image.open(image_path).convert('RGB')
            orig_img = orig_img.resize(self.image_size)
            
            # Extract noise residuals for visualization
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            residual = gray - denoised
            
            # Generate class activation map
            with torch.no_grad():
                if hasattr(self.model, 'features') and hasattr(self.model, 'attention'):
                    # For models with explicit attention mechanism
                    features = self.model.features(img_tensor)
                    attention = self.model.attention(features)
                    
                    # Get activation map from attention
                    activation_map = attention.mean(1).squeeze().cpu().numpy()
                else:
                    # For standard models, use last convolutional layer output
                    # This is a basic approximation for visualization
                    activation_map = np.ones((28, 28))  # Default size
            
            # Resize activation map to match image size
            activation_map = cv2.resize(activation_map, self.image_size)
            
            # Normalize to [0, 1] for visualization
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            
            # Create heatmap
            heatmap = cv2.applyColorMap((activation_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Convert PIL Image to numpy for blending
            orig_np = np.array(orig_img)
            
            # Blend original image with heatmap
            blended = (0.7 * orig_np + 0.3 * heatmap).astype(np.uint8)
            
            # Normalize residual for visualization
            residual_vis = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
            residual_vis = cv2.cvtColor(residual_vis.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            residual_vis = cv2.resize(residual_vis, self.image_size)
            
            # Make predictions for display
            result = self.predict_image(image_path)
            
            # Create figure with subplots
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(orig_np)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(blended)
            plt.title(f"Attention Map\nPrediction: {'STEGO' if result['is_stego'] else 'CLEAN'} ({result['confidence']:.2f})")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(residual_vis)
            plt.title("Noise Residual")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            logger.info(f"Visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return None
    
    def evaluate_dataset(self, test_dir=None, is_stego=False):
        """Evaluate model performance on a dataset"""
        if not test_dir:
            test_dir = self.stego_dir if is_stego else self.cover_dir
        
        if not os.path.exists(test_dir):
            logger.error(f"Test directory not found: {test_dir}")
            return {
                'accuracy': 0.0,
                'total_images': 0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'error_count': 0
            }
        
        image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            logger.warning(f"No images found in directory: {test_dir}")
            return {
                'accuracy': 0.0,
                'total_images': 0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'error_count': 0
            }
        
        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_images': len(image_files),
            'error_count': 0,
            'confidence_scores': []
        }
        
        for img_path in image_files:
            try:
                result = self.predict_image(img_path)
                predicted_stego = result['is_stego']
                metrics['confidence_scores'].append(result['confidence'])
                
                # Update confusion matrix
                if is_stego:  # Expected stego
                    if predicted_stego:
                        metrics['true_positives'] += 1
                    else:
                        metrics['false_negatives'] += 1
                else:  # Expected clean
                    if predicted_stego:
                        metrics['false_positives'] += 1
                    else:
                        metrics['true_negatives'] += 1
                        
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                metrics['error_count'] += 1
        
        # Calculate performance metrics
        correct = metrics['true_positives'] + metrics['true_negatives']
        total = metrics['total_images'] - metrics['error_count']
        
        if total > 0:
            metrics['accuracy'] = correct / total
        else:
            metrics['accuracy'] = 0.0
        
        # Calculate precision
        if metrics['true_positives'] + metrics['false_positives'] > 0:
            metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        else:
            metrics['precision'] = 0.0
        
        # Calculate recall
        if metrics['true_positives'] + metrics['false_negatives'] > 0:
            metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        else:
            metrics['recall'] = 0.0
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        # Calculate average confidence
        if metrics['confidence_scores']:
            metrics['avg_confidence'] = sum(metrics['confidence_scores']) / len(metrics['confidence_scores'])
        else:
            metrics['avg_confidence'] = 0.0
        
        logger.info(f"Evaluation on {'stego' if is_stego else 'cover'} data: "
                   f"Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics

def predict_image(image_path):
    """Convenience function for single image prediction"""
    analyser = CNNAnalyser()
    return analyser.predict_image(image_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CNN-based steganalysis for images')
    parser.add_argument('--file', help='Path to image file for analysis')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--report', '-r', help='Generate report file')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create visualization')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                      help='Detection threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate on test directories')
    parser.add_argument('--model', help='Path to saved model')
    
    args = parser.parse_args()
    
    analyser = CNNAnalyser()
    analyser.detection_threshold = args.threshold
    
    if args.batch_size:
        analyser.batch_size = args.batch_size
    
    # Load pre-trained model if specified
    if args.model and os.path.exists(args.model):
        analyser.load_model(args.model)
    
    if args.train:
        print(f"Training model for {args.epochs} epochs...")
        analyser.train_model(epochs=args.epochs)
        
    elif args.evaluate:
        print("Evaluating on test directories...")
        
        cover_metrics = analyser.evaluate_dataset(COVER_DIR, is_stego=False)
        print("\nCover Images (Expected: Clean):")
        print(f"Accuracy:       {cover_metrics['accuracy']*100:.2f}%")
        print(f"True Negatives: {cover_metrics['true_negatives']}")
        print(f"False Positives: {cover_metrics['false_positives']}")
        print(f"Total Images:   {cover_metrics['total_images']}")
        
        stego_metrics = analyser.evaluate_dataset(STEGO_DIR, is_stego=True)
        print("\nStego Images (Expected: Stego):")
        print(f"Accuracy:       {stego_metrics['accuracy']*100:.2f}%")
        print(f"True Positives: {stego_metrics['true_positives']}")
        print(f"False Negatives: {stego_metrics['false_negatives']}")
        print(f"Total Images:   {stego_metrics['total_images']}")
        
        total_correct = cover_metrics['true_negatives'] + stego_metrics['true_positives']
        total_images = cover_metrics['total_images'] + stego_metrics['total_images']
        overall_accuracy = total_correct / total_images if total_images > 0 else 0
        
        print("\nOverall Performance:")
        print(f"Accuracy:       {overall_accuracy*100:.2f}%")
        print(f"Precision:      {stego_metrics['precision']*100:.2f}%")
        print(f"Recall:         {stego_metrics['recall']*100:.2f}%")
        print(f"F1 Score:       {stego_metrics['f1_score']*100:.2f}%")
        
        # Save evaluation results
        eval_results = {
            "timestamp": datetime.now().isoformat(),
            "cover_metrics": cover_metrics,
            "stego_metrics": stego_metrics,
            "overall_accuracy": overall_accuracy
        }
        
        with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        
    elif args.file:
        print(f"Analyzing {args.file}...")
        result = analyser.predict_image(args.file)
        
        print("\nAnalysis Results:")
        if result['is_stego']:
            print(f"RESULT: STEGANOGRAPHIC CONTENT DETECTED")
        else:
            print(f"RESULT: NO STEGANOGRAPHIC CONTENT DETECTED")
            
        print(f"Confidence:      {result['confidence']*100:.2f}%")
        print(f"Processing time: {result['detection_time']:.3f} seconds")
        
        if args.visualize:
            vis_path = analyser.visualize_features(args.file)
            print(f"Visualization saved to: {vis_path}")
            
        if args.report:
            report_path = args.report if args.report.endswith('.json') else args.report + '.json'
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Detailed report saved to: {report_path}")
    else:
        print("Please specify an action: --train, --evaluate or --file")