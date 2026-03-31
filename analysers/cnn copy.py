import os
import numpy as np
import cv2
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json


COVER_DIR = "data/datasets/cover"  # Directory for clean images
STEGO_DIR = "data/datasets/stego"  # Directory for steganographic images
RESULTS_DIR = "results"   # Directory for storing results

os.makedirs(RESULTS_DIR, exist_ok=True)

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
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        self.model = self._initialize_model()
        self.baseline_stats = self._calculate_baseline_stats()
        
        logger.info(f"CNN Analyser initialized successfully. GPU available: {self.use_gpu}")
    
    def _initialize_model(self):
        try:
            model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 2)  # 2 classes: cover(0), stego(1)
        )
            
            model.eval()
            model = model.to(self.device)
            
            logger.info("Model initialized successfully with MobileNetV2 base")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return self._create_backup_model()
    
    def _create_backup_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        model.eval()
        model = model.to(self.device)
        
        logger.info("Initialized backup model for feature extraction")
        return model
    
    def _calculate_baseline_stats(self):
        baseline_stats = {
            'cover': {
                'feature_mean': 0.0,
                'feature_std': 0.0,
                'noise_kurtosis': 0.0,
                'noise_entropy': 0.0
            },
            'stego': {
                'feature_mean': 0.0,
                'feature_std': 0.0,
                'noise_kurtosis': 0.0,
                'noise_entropy': 0.0
            }
        }
        
        try:
            if not os.path.exists(self.cover_dir):
                logger.warning(f"Cover directory not found: {self.cover_dir}")
                return baseline_stats
                
            if not os.path.exists(self.stego_dir):
                logger.warning(f"Stego directory not found: {self.stego_dir}")
                return baseline_stats
            
            cover_files = [os.path.join(self.cover_dir, f) for f in os.listdir(self.cover_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:5]
                         
            stego_files = [os.path.join(self.stego_dir, f) for f in os.listdir(self.stego_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:5]
            
            if cover_files:
                cover_features = []
                cover_noise_stats = []
                
                for img_path in cover_files:
                    try:
                        img_tensor = self.preprocess_image(img_path)
                        features = self.extract_features(img_tensor)[0]
                        cover_features.append({
                            'mean': np.mean(features),
                            'std': np.std(features)
                        })
                        
                        noise_features = self.extract_noise_residuals(img_path)
                        if noise_features:
                            cover_noise_stats.append({
                                'kurtosis': noise_features.get('kurtosis', 0.0),
                                'entropy': noise_features.get('histogram_entropy', 0.0)
                            })
                    except Exception as e:
                        logger.error(f"Error processing cover image {img_path}: {str(e)}")
                
                if cover_features:
                    baseline_stats['cover']['feature_mean'] = np.mean([f['mean'] for f in cover_features])
                    baseline_stats['cover']['feature_std'] = np.mean([f['std'] for f in cover_features])
                
                if cover_noise_stats:
                    baseline_stats['cover']['noise_kurtosis'] = np.mean([n['kurtosis'] for n in cover_noise_stats])
                    baseline_stats['cover']['noise_entropy'] = np.mean([n['entropy'] for n in cover_noise_stats])
            
            if stego_files:
                stego_features = []
                stego_noise_stats = []
                
                for img_path in stego_files:
                    try:
                        img_tensor = self.preprocess_image(img_path)
                        features = self.extract_features(img_tensor)[0]
                        stego_features.append({
                            'mean': np.mean(features),
                            'std': np.std(features)
                        })
                        
                        noise_features = self.extract_noise_residuals(img_path)
                        if noise_features:
                            stego_noise_stats.append({
                                'kurtosis': noise_features.get('kurtosis', 0.0),
                                'entropy': noise_features.get('histogram_entropy', 0.0)
                            })
                    except Exception as e:
                        logger.error(f"Error processing stego image {img_path}: {str(e)}")
                
                if stego_features:
                    baseline_stats['stego']['feature_mean'] = np.mean([f['mean'] for f in stego_features])
                    baseline_stats['stego']['feature_std'] = np.mean([f['std'] for f in stego_features])
                
                if stego_noise_stats:
                    baseline_stats['stego']['noise_kurtosis'] = np.mean([n['kurtosis'] for n in stego_noise_stats])
                    baseline_stats['stego']['noise_entropy'] = np.mean([n['entropy'] for n in stego_noise_stats])
            
            logger.info("Baseline statistics calculated successfully")
            return baseline_stats
            
        except Exception as e:
            logger.error(f"Error calculating baseline statistics: {str(e)}")
            return baseline_stats
    
    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def extract_features(self, img_tensor):
        try:
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
                
                if not features.shape:
                    features = np.array([features])
                    
                if len(features.shape) > 1:
                    features = features.reshape(-1)
                    
                return np.expand_dims(features, 0)
                
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.random.rand(1, 1280)  # MobileNetV2 feature size
    
    def extract_noise_residuals(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            residual = gray - denoised
            
            mean = np.mean(residual)
            std = np.std(residual)
            skewness = np.mean(((residual - mean) / std) ** 3) if std > 0 else 0
            kurtosis = np.mean(((residual - mean) / std) ** 4) if std > 0 else 0
            
            hist = cv2.calcHist([residual], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            return {
                'mean': float(mean),
                'std': float(std),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'histogram_entropy': float(-np.sum(hist * np.log2(hist + 1e-10))),
                'max_bin': float(np.argmax(hist)),
                'histogram_peaks': int(np.sum(hist > np.mean(hist) + np.std(hist)))
            }
            
        except Exception as e:
            logger.error(f"Error extracting noise residuals: {str(e)}")
            return {}
    
    def analyze_features(self, features, noise_features):
        baseline = self.baseline_stats
        
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_entropy = -np.sum((features * np.log2(features + 1e-10)))
        
        feature_distribution = {}
        for i in range(10):
            bin_range = (i/10, (i+1)/10)
            bin_count = np.sum((features >= bin_range[0]) & (features < bin_range[1]))
            feature_distribution[f"bin_{i}"] = float(bin_count / features.size)
        
        if noise_features:
            noise_score = 0.0
            
            if baseline['cover']['noise_kurtosis'] > 0:
                kurtosis_diff = abs(noise_features['kurtosis'] - baseline['cover']['noise_kurtosis'])
                kurtosis_score = min(0.3, kurtosis_diff / 5.0)
                noise_score += kurtosis_score
                
                entropy_diff = abs(noise_features['histogram_entropy'] - baseline['cover']['noise_entropy'])
                entropy_score = min(0.2, entropy_diff / 2.0)
                noise_score += entropy_score
            else:
                if noise_features['kurtosis'] > 5.0:
                    noise_score += 0.2
                
                if noise_features['histogram_entropy'] < 6.0:
                    noise_score += 0.15
            
            if noise_features['histogram_peaks'] > 5:
                noise_score += 0.15
        else:
            noise_score = 0.0
        
        feature_score = 0.0
        
        if baseline['cover']['feature_mean'] > 0:
            mean_diff = abs(feature_mean - baseline['cover']['feature_mean'])
            std_diff = abs(feature_std - baseline['cover']['feature_std'])
            
            mean_score = min(0.25, mean_diff / baseline['cover']['feature_mean'])
            std_score = min(0.25, std_diff / baseline['cover']['feature_std'])
            
            feature_score += mean_score + std_score
            
            if baseline['stego']['feature_mean'] > 0:
                stego_mean_diff = abs(feature_mean - baseline['stego']['feature_mean'])
                stego_std_diff = abs(feature_std - baseline['stego']['feature_std'])
                
                if stego_mean_diff < mean_diff:
                    feature_score += 0.1
                if stego_std_diff < std_diff:
                    feature_score += 0.1
        else:
            std_ratio = feature_std / (feature_mean + 1e-10)
            if std_ratio > 2.0 or std_ratio < 0.3:
                feature_score += 0.1
            
            if feature_entropy < 4.0:
                feature_score += 0.15
        
        bin_values = list(feature_distribution.values())
        bin_diffs = [abs(bin_values[i] - bin_values[i-1]) for i in range(1, len(bin_values))]
        if max(bin_diffs) > 0.2:
            feature_score += 0.15
        
        stego_score = noise_score * 0.6 + feature_score * 0.4
        confidence = min(0.95, stego_score * 2.0)
        
        return {
            'confidence': confidence,
            'noise_score': noise_score,
            'feature_score': feature_score,
            'feature_mean': float(feature_mean),
            'feature_std': float(feature_std),
            'feature_entropy': float(feature_entropy),
            'feature_distribution': feature_distribution,
            'noise_features': noise_features,
            'baseline_comparison': {
                'cover_mean_diff': float(abs(feature_mean - baseline['cover']['feature_mean'])) if baseline['cover']['feature_mean'] > 0 else 0,
                'cover_noise_diff': float(abs(noise_features.get('kurtosis', 0) - baseline['cover']['noise_kurtosis'])) if baseline['cover']['noise_kurtosis'] > 0 and noise_features else 0
            }
        }
    
    def predict_image(self, image_path):
        logger.info(f"Starting CNN-based steganalysis for: {image_path}")
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
            img_tensor = self.preprocess_image(image_path)
            features = self.extract_features(img_tensor)
            noise_features = self.extract_noise_residuals(image_path)
            analysis_results = self.analyze_features(features[0], noise_features)
            
            confidence = analysis_results['confidence']
            is_stego = confidence > self.detection_threshold
            
            self.detection_result.update({
                'is_stego': is_stego,
                'confidence': confidence,
                'features_analysis': analysis_results,
                'method_used': 'CNN Deep Learning'
            })
            
            detection_time = time.time() - start_time
            self.detection_result['detection_time'] = detection_time
            
            logger.info(f"Analysis complete. Result: {'STEGO' if is_stego else 'CLEAN'} " +
                       f"with {confidence:.2f} confidence " +
                       f"(time: {detection_time:.2f}s)")
            
            return self.detection_result
            
        except Exception as e:
            logger.error(f"Error in CNN steganalysis: {str(e)}")
            
            self.detection_result['detection_time'] = time.time() - start_time
            self.detection_result['error'] = str(e)
            
            return self.detection_result
    
    def visualize_features(self, image_path, output_path=None):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 3, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(2, 3, 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            residual = gray - denoised
            plt.imshow(residual, cmap='jet')
            plt.title("Noise Residual")
            plt.axis("off")
            
            plt.subplot(2, 3, 3)
            hist = cv2.calcHist([residual], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.title("Noise Histogram")
            plt.tight_layout()
            
            plt.subplot(2, 3, 4)
            edges = cv2.Canny(gray, 100, 200)
            plt.imshow(edges, cmap='gray')
            plt.title("Edge Detection")
            plt.axis("off")
            
            plt.subplot(2, 3, 5)
            chans = cv2.split(img)
            colors = ("b", "g", "r")
            for (chan, color) in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Color Histograms")
            
            plt.subplot(2, 3, 6)
            lsb = np.bitwise_and(gray, 1) * 255
            plt.imshow(lsb, cmap='gray')
            plt.title("LSB Plane")
            plt.axis("off")
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return output_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None

    def evaluate_dataset(self, test_dir=None, is_stego=False):
        if not test_dir:
            test_dir = self.stego_dir if is_stego else self.cover_dir
        
        if not os.path.exists(test_dir):
            logger.error(f"Test directory not found: {test_dir}")
            return {
                'accuracy': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'total_images': 0,
                'error_count': 0
            }
        
        image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            logger.warning(f"No images found in directory: {test_dir}")
            return {
                'accuracy': 0.0,
                'total_images': 0
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
                predicted_stego = result.get('is_stego', False)
                confidence = result.get('confidence', 0.0)
                
                metrics['confidence_scores'].append(confidence)
                
                if is_stego and predicted_stego:
                    metrics['true_positives'] += 1
                elif is_stego and not predicted_stego:
                    metrics['false_negatives'] += 1
                elif not is_stego and predicted_stego:
                    metrics['false_positives'] += 1
                else:  # not is_stego and not predicted_stego
                    metrics['true_negatives'] += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating image {img_path}: {str(e)}")
                metrics['error_count'] += 1
        
        correct = metrics['true_positives'] + metrics['true_negatives']
        total = metrics['total_images'] - metrics['error_count']
        metrics['accuracy'] = correct / total if total > 0 else 0.0
        
        if metrics['true_positives'] + metrics['false_positives'] > 0:
            metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        else:
            metrics['precision'] = 0.0
            
        if metrics['true_positives'] + metrics['false_negatives'] > 0:
            metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        else:
            metrics['recall'] = 0.0
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        metrics['avg_confidence'] = np.mean(metrics['confidence_scores']) if metrics['confidence_scores'] else 0.0
        
        logger.info(f"Evaluation complete on {test_dir}. Accuracy: {metrics['accuracy']:.4f}, " +
                  f"TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, " +
                  f"TN: {metrics['true_negatives']}, FN: {metrics['false_negatives']}")
        
        return metrics

def predict_image(image_path):
    analyser = CNNAnalyser()
    return analyser.predict_image(image_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CNN-based steganalysis for images')
    parser.add_argument('file', help='Path to image file for analysis')
    parser.add_argument('--report', '-r', help='Generate report file')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create visualization')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                      help='Detection threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate on test directories')
    
    args = parser.parse_args()
    
    analyser = CNNAnalyser()
    analyser.detection_threshold = args.threshold
    
    if args.evaluate:
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
        
    else:
        print(f"Analyzing {args.file}...")
        result = analyser.predict_image(args.file)
        
        print("\nAnalysis Results:")
        if result['is_stego']:
            print(f"RESULT:          STEGANOGRAPHIC CONTENT DETECTED")
        else:
            print(f"RESULT:          NO STEGANOGRAPHIC CONTENT DETECTED")
            
        print(f"Confidence:      {result['confidence']*100:.2f}%")
        print(f"Processing time: {result['detection_time']:.3f} seconds")
        
        if args.visualize:
            vis_path = os.path.join(RESULTS_DIR, os.path.basename(args.file) + "_visualization.png")
            analyser.visualize_features(args.file, vis_path)
            print(f"\nVisualization saved to: {vis_path}")        