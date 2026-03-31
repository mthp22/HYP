import os
import time
from PIL import Image
import tempfile
import numpy as np

try:
    from analysers.lsboptimised import detect_stego as lsb_detect
except ImportError:
    print("Warning: LSB analyser not available")
    lsb_detect = None

try:
    from analysers.dct import analyze_image as dct_analyze
except ImportError:
    print("Warning: DCT analyser not available")
    dct_analyze = None

try:
    from analysers.cnn import predict_image as cnn_predict
except ImportError:
    print("Warning: CNN analyser not available")
    cnn_predict = None

class FormatConverter:
    
    def __init__(self):
        self.temp_files = []
    
    def convert_to_bmp(self, image_path):
        try:
            if image_path.lower().endswith('.bmp'):
                return image_path
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.bmp', delete=False)
                temp_file.close()
                
                img.save(temp_file.name, 'BMP')
                self.temp_files.append(temp_file.name)
                
                return temp_file.name
                
        except Exception as e:
            print(f"Error converting to BMP: {e}")
            return None
    
    def convert_to_png(self, image_path):
        try:
            if image_path.lower().endswith('.png'):
                return image_path
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_file.close()
                
                img.save(temp_file.name, 'PNG')
                self.temp_files.append(temp_file.name)
                
                return temp_file.name
                
        except Exception as e:
            print(f"Error converting to PNG: {e}")
            return None
    
    def convert_to_jpg(self, image_path):
        try:
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                return image_path
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_file.close()
                
                img.save(temp_file.name, 'JPEG', quality=95)
                self.temp_files.append(temp_file.name)
                
                return temp_file.name
                
        except Exception as e:
            print(f"Error converting to JPG: {e}")
            return None
    
    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_file}: {e}")
        self.temp_files.clear()

class SteganalysisPipeline:
    def __init__(self):
        self.methods = {}
        self.quick_methods = {}
        self.converter = FormatConverter()
        
        # Add available methods
        if lsb_detect:
            self.methods['LSB'] = lsb_detect
        
        if dct_analyze:
            self.methods['DCT'] = dct_analyze
        
        if cnn_predict:
            self.methods['CNN'] = cnn_predict
        
        # Method format requirements
        self.format_requirements = {
            'LSB': 'bmp',
            'DCT': 'png', 
            'CNN': 'png'
        }
        
    def _get_converted_path(self, method_name, original_path):
        required_format = self.format_requirements.get(method_name, 'png')
        
        if required_format == 'bmp':
            return self.converter.convert_to_bmp(original_path)
        elif required_format == 'png':
            return self.converter.convert_to_png(original_path)
        elif required_format == 'jpg':
            return self.converter.convert_to_jpg(original_path)
        else:
            return original_path
   
    def run_analysis(self, method_name, file_path, quick_mode=False):    
            if method_name not in self.methods:
                raise ValueError(f"Unsupported method: {method_name}")
            
            converted_path = self._get_converted_path(method_name, file_path)
            if not converted_path:
                return {
                    'method': method_name,
                    'file_path': file_path,
                    'error': f'Failed to convert image to required format ({self.format_requirements.get(method_name, "unknown")})',
                    'is_stego': False,
                    'confidence': 0,
                    'execution_time': 0
                }
            
            try:
                if quick_mode and method_name in self.quick_methods:
                    start_time = time.time()
                    classification, confidence = self.quick_methods[method_name](converted_path)
                    end_time = time.time()
                    
                    return {
                        'method': method_name,
                        'file_path': file_path,
                        'classification': classification,
                        'confidence': confidence,
                        'is_stego': classification == 'stego',
                        'execution_time': end_time - start_time,
                        'quick_mode': True,
                        'converted_from': os.path.splitext(file_path)[1].lower()
                    }
                else:
                    start_time = time.time()
                    result = self.methods[method_name](converted_path)
                    end_time = time.time()
                    
                    # Debug logging - remove after fixing
                    if method_name in ['DCT', 'CNN']:
                        print(f"\n=== DEBUG: {method_name} Raw Result ===")
                        print(f"Type: {type(result)}")
                        print(f"Content: {result}")
                        print("=" * 40)
                    
                    # Standardize result format
                    standardized_result = self._standardize_result(result, method_name, file_path)
                    standardized_result['execution_time'] = end_time - start_time
                    standardized_result['quick_mode'] = False
                    standardized_result['converted_from'] = os.path.splitext(file_path)[1].lower()
                    
                    # Debug logging for standardized result
                    if method_name in ['DCT', 'CNN']:
                        print(f"\n=== DEBUG: {method_name} Standardized Result ===")
                        print(f"Classification: {standardized_result.get('classification')}")
                        print(f"Is Stego: {standardized_result.get('is_stego')}")
                        print(f"Confidence: {standardized_result.get('confidence')}")
                        print("=" * 50)
                    
                    return standardized_result
                    
            except Exception as e:
                return {
                    'method': method_name,
                    'file_path': file_path,
                    'error': str(e),
                    'is_stego': False,
                    'confidence': 0,
                    'execution_time': 0
                }        
    
    def run_all_methods(self, file_path, quick_mode=False):
        results = {}
        
        for method_name in self.methods.keys():
            try:
                result = self.run_analysis(method_name, file_path, quick_mode)
                results[method_name] = result
            except Exception as e:
                results[method_name] = {
                    'method': method_name,
                    'error': str(e),
                    'is_stego': False,
                    'confidence': 0,
                    'execution_time': 0
                }
        
        return results
    
    def _standardize_result(self, result, method_name, file_path):
        standardized = {
            'method': method_name,
            'file_path': file_path,
            'file_name': os.path.basename(file_path)
        }
        
        if isinstance(result, dict):          
            # LSB analyser format
            if 'is_stego' in result:
                standardized['is_stego'] = result['is_stego']
                standardized['confidence'] = result.get('confidence', 0)
                standardized['classification'] = result.get('classification', 'unknown')
                
                # LSB-specific fields
                if method_name == 'LSB':
                    standardized['entropy'] = result.get('entropy', 0)
                    if 'bit_distribution' in result and 'combined_lsbs' in result['bit_distribution']:
                        standardized['chi_square'] = result['bit_distribution']['combined_lsbs'].get('chi_square', 0)
                    if 'runs_test' in result:
                        standardized['z_score'] = result['runs_test'].get('z_score', 0)
            
            # DCT analyser format - Updated handling
            elif 'result' in result and 'confidence' in result:
                result_value = result['result']
                # Handle different possible result formats
                if isinstance(result_value, str):
                    is_stego = result_value.upper() in ['STEGO', 'STEGANOGRAPHY', 'SUSPICIOUS']
                    classification = 'stego' if is_stego else 'clean'
                elif isinstance(result_value, bool):
                    is_stego = result_value
                    classification = 'stego' if is_stego else 'clean'
                else:
                    # Handle numeric results (e.g., 0/1, probability)
                    is_stego = float(result_value) > 0.5 if result_value is not None else False
                    classification = 'stego' if is_stego else 'clean'
                
                standardized['is_stego'] = is_stego
                standardized['confidence'] = result['confidence']
                standardized['classification'] = classification
                
                # DCT-specific fields
                if method_name == 'DCT':
                    standardized['anomaly_score'] = result.get('anomaly_score', 0)
                    standardized['dct_entropy'] = result.get('entropy', 0)
            
            # CNN analyser format - Updated handling
            elif 'prediction' in result:
                prediction = result['prediction']
                # Handle different prediction formats
                if isinstance(prediction, str):
                    is_stego = prediction.upper() in ['STEGO', 'STEGANOGRAPHY', 'SUSPICIOUS']
                    classification = prediction.lower() if prediction.lower() in ['stego', 'clean'] else ('stego' if is_stego else 'clean')
                elif isinstance(prediction, bool):
                    is_stego = prediction
                    classification = 'stego' if is_stego else 'clean'
                else:
                    # Handle numeric predictions
                    is_stego = float(prediction) > 0.5 if prediction is not None else False
                    classification = 'stego' if is_stego else 'clean'
                
                standardized['is_stego'] = is_stego
                standardized['confidence'] = result.get('confidence', 0)
                standardized['classification'] = classification
                
                # CNN-specific fields
                if method_name == 'CNN':
                    standardized['prediction_probabilities'] = result.get('probabilities', {})
                    standardized['feature_analysis'] = result.get('feature_analysis', {})
            
            # Handle additional possible formats for DCT/CNN
            elif 'classification' in result or 'label' in result or 'class' in result:
                # Direct classification field
                classification_field = result.get('classification') or result.get('label') or result.get('class')
                
                if isinstance(classification_field, str):
                    classification = classification_field.lower()
                    is_stego = classification in ['stego', 'steganography', 'suspicious']
                else:
                    is_stego = bool(classification_field)
                    classification = 'stego' if is_stego else 'clean'
                
                standardized['is_stego'] = is_stego
                standardized['confidence'] = result.get('confidence', result.get('score', 0.5))
                standardized['classification'] = classification
            
            # Handle probability-based results
            elif 'probability' in result or 'score' in result:
                probability = result.get('probability', result.get('score', 0.5))
                is_stego = probability > 0.5
                classification = 'stego' if is_stego else 'clean'
                
                standardized['is_stego'] = is_stego
                standardized['confidence'] = probability
                standardized['classification'] = classification
            
            # Generic format fallback - Enhanced
            else:
                # Try to infer from available fields with better logic
                confidence = result.get('confidence', result.get('score', result.get('probability', 0)))
                
                # Look for classification indicators with more comprehensive checking
                classification = 'unknown'
                is_stego = False
                
                # Check various possible field names
                for field in ['classification', 'result', 'prediction', 'label', 'class', 'output']:
                    if field in result:
                        field_value = result[field]
                        if isinstance(field_value, str):
                            field_lower = field_value.lower()
                            if field_lower in ['stego', 'steganography', 'suspicious', 'hidden']:
                                classification = 'stego'
                                is_stego = True
                                break
                            elif field_lower in ['clean', 'normal', 'original', 'cover']:
                                classification = 'clean'
                                is_stego = False
                                break
                        elif isinstance(field_value, (int, float)):
                            # Numeric classification
                            if field_value > 0.5:
                                classification = 'stego'
                                is_stego = True
                            else:
                                classification = 'clean'
                                is_stego = False
                            break
                        elif isinstance(field_value, bool):
                            is_stego = field_value
                            classification = 'stego' if is_stego else 'clean'
                            break
                
                # If still unknown, use confidence threshold
                if classification == 'unknown' and confidence > 0:
                    is_stego = confidence > 0.5
                    classification = 'stego' if is_stego else 'clean'
                
                standardized['is_stego'] = is_stego
                standardized['confidence'] = confidence
                standardized['classification'] = classification
        
        else:
            # Handle simple return values (boolean, string, number)
            if isinstance(result, bool):
                is_stego = result
                classification = 'stego' if result else 'clean'
                confidence = 0.8 if result else 0.2
            elif isinstance(result, str):
                result_lower = result.lower()
                if result_lower in ['stego', 'steganography', 'suspicious']:
                    is_stego = True
                    classification = 'stego'
                    confidence = 0.8
                elif result_lower in ['clean', 'normal', 'cover']:
                    is_stego = False
                    classification = 'clean'
                    confidence = 0.8
                else:
                    is_stego = False
                    classification = 'unknown'
                    confidence = 0.5
            elif isinstance(result, (int, float)):
                is_stego = result > 0.5
                classification = 'stego' if is_stego else 'clean'
                confidence = float(result)
            else:
                # Fallback for any other type
                is_stego = bool(result)
                classification = 'stego' if is_stego else 'clean'
                confidence = 0.7 if result else 0.3
            
            standardized['is_stego'] = is_stego
            standardized['confidence'] = confidence
            standardized['classification'] = classification
        
        return standardized

    def compare_results(self, results_list):
        if not results_list:
            return {}
        
        if isinstance(results_list, dict):
            methods_results = results_list
        else:
            methods_results = {result['method']: result for result in results_list if 'method' in result}
        
        comparison = {
            'total_methods': len(methods_results),
            'methods_used': list(methods_results.keys()),
            'individual_results': methods_results,
            'consensus_analysis': self._analyze_consensus(methods_results),
            'performance_analysis': self._analyze_performance(methods_results),
            'ensemble_prediction': self._ensemble_predict(methods_results)
        }
        
        return comparison
    
    def _analyze_consensus(self, methods_results):
        valid_results = {k: v for k, v in methods_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'consensus': 'no_valid_results', 'agreement_score': 0}
        
        stego_votes = sum(1 for result in valid_results.values() if result.get('is_stego', False))
        total_votes = len(valid_results)
        
        weighted_score = 0
        total_weight = 0
        
        for result in valid_results.values():
            confidence = result.get('confidence', 0)
            is_stego = result.get('is_stego', False)
            
            weight = confidence
            vote_value = 1 if is_stego else 0
            
            weighted_score += vote_value * weight
            total_weight += weight
        
        weighted_consensus = weighted_score / total_weight if total_weight > 0 else 0
        
        if stego_votes / total_votes >= 0.6:
            consensus = 'stego'
        elif stego_votes / total_votes <= 0.4:
            consensus = 'clean'
        else:
            consensus = 'uncertain'
        
        agreement_score = max(stego_votes, total_votes - stego_votes) / total_votes
        
        return {
            'consensus': consensus,
            'agreement_score': agreement_score,
            'stego_votes': stego_votes,
            'total_votes': total_votes,
            'weighted_consensus': weighted_consensus,
            'unanimous': stego_votes == 0 or stego_votes == total_votes
        }
    
    def _analyze_performance(self, methods_results):
        valid_results = {k: v for k, v in methods_results.items() if 'error' not in v}
        
        performance = {}
        
        for method_name, result in valid_results.items():
            execution_time = result.get('execution_time', 0)
            confidence = result.get('confidence', 0)
            
            performance[method_name] = {
                'execution_time': execution_time,
                'confidence': confidence,
                'classification': result.get('classification', 'unknown'),
                'is_stego': result.get('is_stego', False),
                'quick_mode': result.get('quick_mode', False),
                'converted_from': result.get('converted_from', 'unknown')
            }
        
        if valid_results:
            fastest_method = min(performance.keys(), 
                               key=lambda x: performance[x]['execution_time'])
            most_confident = max(performance.keys(), 
                               key=lambda x: performance[x]['confidence'])
            
            total_time = sum(perf['execution_time'] for perf in performance.values())
            avg_confidence = sum(perf['confidence'] for perf in performance.values()) / len(performance)
            
            performance['summary'] = {
                'fastest_method': fastest_method,
                'fastest_time': performance[fastest_method]['execution_time'],
                'most_confident_method': most_confident,
                'highest_confidence': performance[most_confident]['confidence'],
                'total_execution_time': total_time,
                'average_confidence': avg_confidence
            }
        
        return performance
    
    def _ensemble_predict(self, methods_results):
        valid_results = {k: v for k, v in methods_results.items() if 'error' not in v}
        
        if not valid_results:
            return {
                'ensemble_classification': 'error',
                'ensemble_confidence': 0,
                'ensemble_is_stego': False,
                'method': 'ensemble'
            }
        
        method_weights = {
            'LSB': 0.4,   
            'DCT': 0.35,  
            'CNN': 0.25   
        }
        
        weighted_votes = 0
        total_weight = 0
        confidence_scores = []
        
        for method_name, result in valid_results.items():
            method_weight = method_weights.get(method_name, 0.2)
            confidence = result.get('confidence', 0)
            is_stego = result.get('is_stego', False)
            
            adjusted_weight = method_weight * (0.7 + 0.3 * confidence)
            
            vote_value = 1 if is_stego else 0
            weighted_votes += vote_value * adjusted_weight
            total_weight += adjusted_weight
            confidence_scores.append(confidence)
        
        # Calculate ensemble prediction
        ensemble_score = weighted_votes / total_weight if total_weight > 0 else 0
        ensemble_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Apply ensemble confidence boost for agreement
        consensus_analysis = self._analyze_consensus(valid_results)
        if consensus_analysis['unanimous']:
            ensemble_confidence = min(ensemble_confidence * 1.2, 1.0)
        elif consensus_analysis['agreement_score'] > 0.8:
            ensemble_confidence = min(ensemble_confidence * 1.1, 1.0)
        
        # Determine final classification with adjusted thresholds
        if ensemble_score > 0.6:
            ensemble_classification = 'stego'
            ensemble_is_stego = True
        elif ensemble_score < 0.4:
            ensemble_classification = 'clean' 
            ensemble_is_stego = False
        else:
            ensemble_classification = 'uncertain'
            ensemble_is_stego = ensemble_score > 0.5
        
        return {
            'ensemble_classification': ensemble_classification,
            'ensemble_confidence': ensemble_confidence,
            'ensemble_score': ensemble_score,
            'ensemble_is_stego': ensemble_is_stego,
            'method': 'ensemble',
            'contributing_methods': list(valid_results.keys()),
            'method_weights': {k: method_weights.get(k, 0.2) for k in valid_results.keys()}
        }
    
    def batch_analyze(self, directory_path, methods=None, quick_mode=False, output_file=None):
        if methods is None:
            methods = list(self.methods.keys())
        
        if isinstance(methods, str):
            methods = [methods]
        
        image_files = []
        supported_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_files.append(os.path.join(root, file))
        
        print(f"Found {len(image_files)} image files to analyze...")
        print(f"Methods to use: {', '.join(methods)}")
        print(f"Format conversion will be handled automatically")
        
        batch_results = []
        
        try:
            for i, file_path in enumerate(image_files):
                print(f"Analyzing {i+1}/{len(image_files)}: {os.path.basename(file_path)}")
                
                file_results = {}
                
                # Run specified methods
                for method in methods:
                    try:
                        result = self.run_analysis(method, file_path, quick_mode)
                        file_results[method] = result
                    except Exception as e:
                        file_results[method] = {
                            'method': method,
                            'error': str(e),
                            'is_stego': False,
                            'confidence': 0
                        }
                
                # Generate comparison if multiple methods
                if len(methods) > 1:
                    comparison = self.compare_results(file_results)
                    file_results['comparison'] = comparison
                
                batch_results.append({
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'results': file_results
                })
                
                # Print summary for this file
                if len(methods) > 1 and 'comparison' in file_results:
                    ensemble = file_results['comparison']['ensemble_prediction']
                    print(f"  Ensemble: {ensemble['ensemble_classification']} "
                          f"(Confidence: {ensemble['ensemble_confidence']:.3f})")
                else:
                    result = list(file_results.values())[0]
                    if 'error' not in result:
                        print(f"  {methods[0]}: {result.get('classification', 'error')} "
                              f"(Confidence: {result.get('confidence', 0):.3f})")
                    else:
                        print(f"  {methods[0]}: Error - {result['error']}")
        
        finally:
            self.converter.cleanup()
        
        # Save results
        if output_file:
            self._save_batch_results(batch_results, output_file, methods)
        
        return batch_results
    
    def _save_batch_results(self, batch_results, output_file, methods):
        import csv
        
        with open(output_file, 'w', newline='') as csvfile:
            base_columns = ['File', 'OriginalFormat']
            
            if len(methods) > 1:
                base_columns.extend(['EnsembleClassification', 'EnsembleConfidence', 'EnsembleScore'])
            
            method_columns = []
            for method in methods:
                method_columns.extend([
                    f'{method}_Classification',
                    f'{method}_Confidence',
                    f'{method}_IsStego',
                    f'{method}_ExecutionTime'
                ])
            
            if len(methods) > 1:
                consensus_columns = ['ConsensusAgreement', 'UnanimousVote', 'StegoVotes', 'TotalVotes']
            else:
                consensus_columns = []
            
            all_columns = base_columns + method_columns + consensus_columns
            
            writer = csv.writer(csvfile)
            writer.writerow(all_columns)
            
            for batch_result in batch_results:
                row = [batch_result['file_name']]                
                original_format = os.path.splitext(batch_result['file_name'])[1].lower()
                row.append(original_format)
                
                results = batch_result['results']
                
                if len(methods) > 1 and 'comparison' in results:
                    ensemble = results['comparison']['ensemble_prediction']
                    row.extend([
                        ensemble['ensemble_classification'],
                        f"{ensemble['ensemble_confidence']:.4f}",
                        f"{ensemble['ensemble_score']:.4f}"
                    ])
                
                for method in methods:
                    if method in results and 'error' not in results[method]:
                        result = results[method]
                        row.extend([
                            result.get('classification', 'error'),
                            f"{result.get('confidence', 0):.4f}",
                            result.get('is_stego', False),
                            f"{result.get('execution_time', 0):.4f}"
                        ])
                    else:
                        row.extend(['error', '0.0000', False, '0.0000'])
                
                if len(methods) > 1 and 'comparison' in results:
                    consensus = results['comparison']['consensus_analysis']
                    row.extend([
                        f"{consensus['agreement_score']:.4f}",
                        consensus['unanimous'],
                        consensus['stego_votes'],
                        consensus['total_votes']
                    ])
                
                writer.writerow(row)
        
        print(f"Results saved to: {output_file}")
    
    def cleanup(self):
        self.converter.cleanup()
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Single file: python pipeline2.py <method> <image_path> [--quick]")
        print("  All methods: python pipeline2.py all <image_path> [--quick]")
        print("  Batch analysis: python pipeline2.py <method> <directory> [output.csv] [--quick]")
        print("  Available methods: LSB, DCT, CNN, all")
        print("  Supported formats: BMP, PNG, JPG, JPEG, TIFF (auto-converted)")
        return
    
    pipeline = SteganalysisPipeline()
    method = sys.argv[1].upper()
    input_path = sys.argv[2]
    
    quick_mode = '--quick' in sys.argv
    output_file = None
    
    # Check for output file argument
    for arg in sys.argv[3:]:
        if not arg.startswith('--') and arg.endswith('.csv'):
            output_file = arg
            break
    
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist.")
        return
    
    try:
        if os.path.isfile(input_path):
            print(f"Analyzing file: {input_path}")
            
            if method == 'ALL':
                results = pipeline.run_all_methods(input_path, quick_mode)
                comparison = pipeline.compare_results(results)
                
                print("\n=== Individual Results ===")
                for method_name, result in results.items():
                    if 'error' not in result:
                        converted_info = f" (converted from {result.get('converted_from', 'unknown')})" if result.get('converted_from') != '.'+pipeline.format_requirements.get(method_name, 'png') else ""
                        print(f"{method_name}: {result['classification']} (Confidence: {result['confidence']:.3f}){converted_info}")
                    else:
                        print(f"{method_name}: Error - {result['error']}")
                
                print("\n=== Ensemble Prediction ===")
                ensemble = comparison['ensemble_prediction']
                print(f"Classification: {ensemble['ensemble_classification']}")
                print(f"Confidence: {ensemble['ensemble_confidence']:.3f}")
                print(f"Score: {ensemble['ensemble_score']:.3f}")
                print(f"Agreement Score: {comparison['consensus_analysis']['agreement_score']:.3f}")
                
                if comparison['consensus_analysis']['unanimous']:
                    print("✓ Unanimous decision across all methods")
                
            else:
                result = pipeline.run_analysis(method, input_path, quick_mode)
                if 'error' not in result:
                    converted_info = f" (converted from {result.get('converted_from', 'unknown')})" if result.get('converted_from') != '.'+pipeline.format_requirements.get(method, 'png') else ""
                    print(f"Result: {result['classification']} (Confidence: {result['confidence']:.3f}){converted_info}")
                    print(f"Execution Time: {result['execution_time']:.3f}s")
                else:
                    print(f"Error: {result['error']}")
        
        else:
            print(f"Batch analyzing directory: {input_path}")
            
            if method == 'ALL':
                methods = list(pipeline.methods.keys())
            else:
                methods = [method]
            
            batch_results = pipeline.batch_analyze(input_path, methods, quick_mode, output_file)
            
            summary = generate_summary_report(batch_results, methods)
            print(f"\n{summary}")
    
    finally:
        pipeline.cleanup()

def generate_summary_report(batch_results, methods_used):
    if not batch_results:
        return "No results to analyze."
    
    total_files = len(batch_results)
    
    method_stats = {}
    for method in methods_used:
        method_stats[method] = {
            'stego_count': 0,
            'clean_count': 0,
            'error_count': 0,
            'total_time': 0,
            'avg_confidence': 0,
            'confidences': [],
            'conversions': {}
        }
    
    ensemble_stats = {
        'stego_count': 0,
        'clean_count': 0,
        'uncertain_count': 0,
        'unanimous_count': 0,
        'confidences': []
    }
    
    for batch_result in batch_results:
        results = batch_result['results']
        
        # Individual method stats
        for method in methods_used:
            if method in results:
                result = results[method]
                stats = method_stats[method]
                
                if 'error' in result:
                    stats['error_count'] += 1
                else:
                    if result.get('is_stego', False):
                        stats['stego_count'] += 1
                    else:
                        stats['clean_count'] += 1
                    
                    stats['total_time'] += result.get('execution_time', 0)
                    confidence = result.get('confidence', 0)
                    stats['confidences'].append(confidence)
                    
                    # Track format conversions
                    converted_from = result.get('converted_from', 'unknown')
                    if converted_from in stats['conversions']:
                        stats['conversions'][converted_from] += 1
                    else:
                        stats['conversions'][converted_from] = 1
        
        if 'comparison' in results:
            ensemble = results['comparison']['ensemble_prediction']
            consensus = results['comparison']['consensus_analysis']
            
            classification = ensemble['ensemble_classification']
            if classification == 'stego':
                ensemble_stats['stego_count'] += 1
            elif classification == 'clean':
                ensemble_stats['clean_count'] += 1
            else:
                ensemble_stats['uncertain_count'] += 1
            
            if consensus['unanimous']:
                ensemble_stats['unanimous_count'] += 1
            
            ensemble_stats['confidences'].append(ensemble['ensemble_confidence'])
    
    for method, stats in method_stats.items():
        if stats['confidences']:
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            stats['avg_time'] = stats['total_time'] / len(stats['confidences'])
    
    if ensemble_stats['confidences']:
        ensemble_stats['avg_confidence'] = sum(ensemble_stats['confidences']) / len(ensemble_stats['confidences'])
    
    # Generate report
    report = []
    report.append("=== Stegananalysis Batch Report ===\n")
    report.append(f"Total Files Analyzed: {total_files}")
    report.append(f"Methods Used: {', '.join(methods_used)}\n")
    
    # Individual method performance
    report.append("Individual Method Performance:")
    for method, stats in method_stats.items():
        total_analyzed = stats['stego_count'] + stats['clean_count']
        if total_analyzed > 0:
            report.append(f"\n{method}:")
            report.append(f"  Files Analyzed: {total_analyzed}")
            report.append(f"  Stego Detected: {stats['stego_count']} ({stats['stego_count']/total_analyzed*100:.1f}%)")
            report.append(f"  Clean Detected: {stats['clean_count']} ({stats['clean_count']/total_analyzed*100:.1f}%)")
            report.append(f"  Errors: {stats['error_count']}")
            report.append(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
            report.append(f"  Avg Time: {stats.get('avg_time', 0):.3f}s")
            

            if stats['conversions']:
                report.append(f"  Format Conversions:")
                for fmt, count in stats['conversions'].items():
                    report.append(f"    {fmt}: {count} files")
    
    # Ensemble performance (if multiple methods)
    if len(methods_used) > 1:
        report.append(f"\nEnsemble Performance:")
        report.append(f"  Stego Detected: {ensemble_stats['stego_count']} ({ensemble_stats['stego_count']/total_files*100:.1f}%)")
        report.append(f"  Clean Detected: {ensemble_stats['clean_count']} ({ensemble_stats['clean_count']/total_files*100:.1f}%)")
        report.append(f"  Uncertain: {ensemble_stats['uncertain_count']} ({ensemble_stats['uncertain_count']/total_files*100:.1f}%)")
        report.append(f"  Unanimous Decisions: {ensemble_stats['unanimous_count']} ({ensemble_stats['unanimous_count']/total_files*100:.1f}%)")
        if ensemble_stats['confidences']:
            report.append(f"  Avg Ensemble Confidence: {ensemble_stats['avg_confidence']:.3f}")
    
    report.append("\n=== End Report ===")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()