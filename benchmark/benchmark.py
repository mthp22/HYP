# Benchmark Tool: Compare accuracy/time across methods

import os
import sys
import time
import json
import csv
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.pipeline2 import SteganalysisPipeline

class BenchmarkTool:
    def __init__(self, dataset_paths):
        if isinstance(dataset_paths, str):
            # Single directory containing all images
            self.dataset_paths = {'mixed': dataset_paths}
        else:
            # Separate cover and stego directories
            self.dataset_paths = dataset_paths
        
        self.results = []
        self.pipeline = SteganalysisPipeline()
        self.methods = list(self.pipeline.methods.keys())
        
    def run_all_methods_on_dataset(self, quick_mode=False, sample_size=None):
        print("Starting benchmark analysis...")
        print(f"Available methods: {', '.join(self.methods)}")
        
        image_files = self._collect_image_files(sample_size)
        total_files = len(image_files)
        
        if total_files == 0:
            print("No image files found in dataset paths!")
            return
        
        print(f"Found {total_files} image files to analyze")
        
        benchmark_start_time = time.time()
        
        try:
            for i, (file_path, ground_truth) in enumerate(image_files):
                print(f"\nAnalyzing {i+1}/{total_files}: {os.path.basename(file_path)}")
                
                file_result = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'ground_truth': ground_truth,
                    'timestamp': datetime.now().isoformat(),
                    'method_results': {},
                    'ensemble_result': None
                }
                method_results = {}
                for method in self.methods:
                    try:
                        print(f"  Running {method}...")
                        result = self.pipeline.run_analysis(method, file_path, quick_mode)
                        
                        if 'error' not in result:
                            predicted_stego = result.get('is_stego', False)
                            actual_stego = (ground_truth == 'stego')
                            
                            result['accuracy_metrics'] = {
                                'correct_prediction': predicted_stego == actual_stego,
                                'true_positive': predicted_stego and actual_stego,
                                'true_negative': not predicted_stego and not actual_stego,
                                'false_positive': predicted_stego and not actual_stego,
                                'false_negative': not predicted_stego and actual_stego
                            }
                        
                        method_results[method] = result
                        file_result['method_results'][method] = result
                        
                    except Exception as e:
                        error_result = {
                            'method': method,
                            'error': str(e),
                            'execution_time': 0,
                            'accuracy_metrics': {
                                'correct_prediction': False,
                                'true_positive': False,
                                'true_negative': False,
                                'false_positive': False,
                                'false_negative': False
                            }
                        }
                        method_results[method] = error_result
                        file_result['method_results'][method] = error_result
                
                if len([r for r in method_results.values() if 'error' not in r]) > 1:
                    comparison = self.pipeline.compare_results(method_results)
                    ensemble = comparison['ensemble_prediction']
                    
                    # Add accuracy metrics for ensemble
                    predicted_stego = ensemble.get('ensemble_is_stego', False)
                    actual_stego = (ground_truth == 'stego')
                    
                    ensemble['accuracy_metrics'] = {
                        'correct_prediction': predicted_stego == actual_stego,
                        'true_positive': predicted_stego and actual_stego,
                        'true_negative': not predicted_stego and not actual_stego,
                        'false_positive': predicted_stego and not actual_stego,
                        'false_negative': not predicted_stego and actual_stego
                    }
                    
                    file_result['ensemble_result'] = ensemble
                    file_result['comparison_analysis'] = comparison
                
                self.results.append(file_result)
                
                correct_methods = sum(1 for r in method_results.values() 
                                    if 'error' not in r and 
                                    r.get('accuracy_metrics', {}).get('correct_prediction', False))
                print(f"    Methods correct: {correct_methods}/{len(self.methods)}")
                
        finally:
            self.pipeline.cleanup()
        
        benchmark_total_time = time.time() - benchmark_start_time
        print(f"\nBenchmark completed in {benchmark_total_time:.2f} seconds")
        
    def _collect_image_files(self, sample_size=None):
        image_files = []
        supported_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        if 'mixed' in self.dataset_paths:
            directory = self.dataset_paths['mixed']
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(fmt) for fmt in supported_formats):
                        file_path = os.path.join(root, file)
                        
                        ground_truth = self._infer_ground_truth(file_path)
                        image_files.append((file_path, ground_truth))
        else:
            if 'cover' in self.dataset_paths:
                cover_dir = self.dataset_paths['cover']
                for root, dirs, files in os.walk(cover_dir):
                    for file in files:
                        if any(file.lower().endswith(fmt) for fmt in supported_formats):
                            file_path = os.path.join(root, file)
                            image_files.append((file_path, 'clean'))
            
            if 'stego' in self.dataset_paths:
                stego_dir = self.dataset_paths['stego']
                for root, dirs, files in os.walk(stego_dir):
                    for file in files:
                        if any(file.lower().endswith(fmt) for fmt in supported_formats):
                            file_path = os.path.join(root, file)
                            image_files.append((file_path, 'stego'))
        
        if sample_size and len(image_files) > sample_size:
            stego_files = [(f, gt) for f, gt in image_files if gt == 'stego']
            clean_files = [(f, gt) for f, gt in image_files if gt == 'clean']
            
            if stego_files and clean_files:
                half_sample = sample_size // 2
                image_files = stego_files[:half_sample] + clean_files[:half_sample]
            else:
                image_files = image_files[:sample_size]
        
        return image_files
    
    def _infer_ground_truth(self, file_path):
        path_lower = file_path.lower()
        
        #  patterns for images
        stego_indicators = ['stego', 'hidden', 'embed', 'watermark', 'lsb']
        clean_indicators = ['cover', 'original', 'clean', 'normal']
        
        for indicator in stego_indicators:
            if indicator in path_lower:
                return 'stego'
        
        for indicator in clean_indicators:
            if indicator in path_lower:
                return 'clean'
        
        return 'unknown'
    
    def generate_comparison_report(self, output_file=None):
        if not self.results:
            return "No benchmark results available. Run benchmark first."
        
        stats = self._calculate_statistics()
        report = self._generate_text_report(stats)
        
        if output_file:
            if output_file.endswith('.json'):
                self._save_json_report(output_file, stats)
            elif output_file.endswith('.csv'):
                self._save_csv_report(output_file)
            else:
                self._save_text_report(output_file, report)
            
            print(f"Report saved to: {output_file}")
        
        return report
    
    def _calculate_statistics(self):
        """Calculate comprehensive performance statistics."""
        stats = {
            'total_files': len(self.results),
            'methods': {},
            'ensemble': {},
            'overall_summary': {}
        }
        
        # Initialize method stats
        for method in self.methods:
            stats['methods'][method] = {
                'total_analyzed': 0,
                'errors': 0,
                'correct_predictions': 0,
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'execution_times': [],
                'confidences': [],
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        
        # Initialize ensemble stats
        stats['ensemble'] = {
            'total_analyzed': 0,
            'correct_predictions': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'confidences': [],
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
        
        for result in self.results:
            if result['ground_truth'] == 'unknown':
                continue
            
            for method, method_result in result['method_results'].items():
                method_stats = stats['methods'][method]
                method_stats['total_analyzed'] += 1
                
                if 'error' in method_result:
                    method_stats['errors'] += 1
                else:
                    # Extract metrics
                    metrics = method_result.get('accuracy_metrics', {})
                    method_stats['correct_predictions'] += int(metrics.get('correct_prediction', False))
                    method_stats['true_positives'] += int(metrics.get('true_positive', False))
                    method_stats['true_negatives'] += int(metrics.get('true_negative', False))
                    method_stats['false_positives'] += int(metrics.get('false_positive', False))
                    method_stats['false_negatives'] += int(metrics.get('false_negative', False))
                    
                    # Performance metrics
                    method_stats['execution_times'].append(method_result.get('execution_time', 0))
                    method_stats['confidences'].append(method_result.get('confidence', 0))
            
            if result.get('ensemble_result'):
                ensemble_result = result['ensemble_result']
                ensemble_stats = stats['ensemble']
                ensemble_stats['total_analyzed'] += 1
                
                metrics = ensemble_result.get('accuracy_metrics', {})
                ensemble_stats['correct_predictions'] += int(metrics.get('correct_prediction', False))
                ensemble_stats['true_positives'] += int(metrics.get('true_positive', False))
                ensemble_stats['true_negatives'] += int(metrics.get('true_negative', False))
                ensemble_stats['false_positives'] += int(metrics.get('false_positive', False))
                ensemble_stats['false_negatives'] += int(metrics.get('false_negative', False))
                ensemble_stats['confidences'].append(ensemble_result.get('ensemble_confidence', 0))
        
        # Calculate  metrics
        for method, method_stats in stats['methods'].items():
            if method_stats['total_analyzed'] > 0:
                # Accuracy
                method_stats['accuracy'] = method_stats['correct_predictions'] / method_stats['total_analyzed']
                
                # Precision, Recall, F1
                tp = method_stats['true_positives']
                fp = method_stats['false_positives']
                fn = method_stats['false_negatives']
                
                method_stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                method_stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if method_stats['precision'] + method_stats['recall'] > 0:
                    method_stats['f1_score'] = 2 * (method_stats['precision'] * method_stats['recall']) / (method_stats['precision'] + method_stats['recall'])

                if method_stats['execution_times']:
                    method_stats['avg_execution_time'] = sum(method_stats['execution_times']) / len(method_stats['execution_times'])
                if method_stats['confidences']:
                    method_stats['avg_confidence'] = sum(method_stats['confidences']) / len(method_stats['confidences'])
        
        ensemble_stats = stats['ensemble']
        if ensemble_stats['total_analyzed'] > 0:
            ensemble_stats['accuracy'] = ensemble_stats['correct_predictions'] / ensemble_stats['total_analyzed']
            
            tp = ensemble_stats['true_positives']
            fp = ensemble_stats['false_positives']
            fn = ensemble_stats['false_negatives']
            
            ensemble_stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            ensemble_stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if ensemble_stats['precision'] + ensemble_stats['recall'] > 0:
                ensemble_stats['f1_score'] = 2 * (ensemble_stats['precision'] * ensemble_stats['recall']) / (ensemble_stats['precision'] + ensemble_stats['recall'])
            
            if ensemble_stats['confidences']:
                ensemble_stats['avg_confidence'] = sum(ensemble_stats['confidences']) / len(ensemble_stats['confidences'])
        
        return stats
    
    def _generate_text_report(self, stats):
        report = []
        report.append("=" * 80)
        report.append("STEGANANALYSIS BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Files Analyzed: {stats['total_files']}")
        report.append(f"Methods Tested: {', '.join(self.methods)}")
        report.append("")

        report.append("INDIVIDUAL METHOD PERFORMANCE")
        report.append("-" * 50)
        
        for method, method_stats in stats['methods'].items():
            if method_stats['total_analyzed'] > 0:
                report.append(f"\n{method}:")
                report.append(f"  Files Analyzed: {method_stats['total_analyzed']}")
                report.append(f"  Errors: {method_stats['errors']}")
                report.append(f"  Accuracy: {method_stats['accuracy']:.3f} ({method_stats['accuracy']*100:.1f}%)")
                report.append(f"  Precision: {method_stats['precision']:.3f}")
                report.append(f"  Recall: {method_stats['recall']:.3f}")
                report.append(f"  F1-Score: {method_stats['f1_score']:.3f}")
                
                if method_stats.get('avg_execution_time'):
                    report.append(f"  Avg Execution Time: {method_stats['avg_execution_time']:.3f}s")
                if method_stats.get('avg_confidence'):
                    report.append(f"  Avg Confidence: {method_stats['avg_confidence']:.3f}")
        
        ensemble_stats = stats['ensemble']
        if ensemble_stats['total_analyzed'] > 0:
            report.append(f"\nENSEMBLE PERFORMANCE")
            report.append("-" * 50)
            report.append(f"Files Analyzed: {ensemble_stats['total_analyzed']}")
            report.append(f"Accuracy: {ensemble_stats['accuracy']:.3f} ({ensemble_stats['accuracy']*100:.1f}%)")
            report.append(f"Precision: {ensemble_stats['precision']:.3f}")
            report.append(f"Recall: {ensemble_stats['recall']:.3f}")
            report.append(f"F1-Score: {ensemble_stats['f1_score']:.3f}")
            
            if ensemble_stats.get('avg_confidence'):
                report.append(f"Avg Confidence: {ensemble_stats['avg_confidence']:.3f}")
        
        report.append(f"\nPERFORMANCE RANKING")
        report.append("-" * 50)
        
        method_ranking = [(method, stats['methods'][method]['accuracy']) 
                         for method in self.methods 
                         if stats['methods'][method]['total_analyzed'] > 0]
        method_ranking.sort(key=lambda x: x[1], reverse=True)
        
        report.append("By Accuracy:")
        for i, (method, accuracy) in enumerate(method_ranking, 1):
            report.append(f"  {i}. {method}: {accuracy:.3f}")
        
        # Rank by F1-score
        f1_ranking = [(method, stats['methods'][method]['f1_score']) 
                     for method in self.methods 
                     if stats['methods'][method]['total_analyzed'] > 0]
        f1_ranking.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\nBy F1-Score:")
        for i, (method, f1_score) in enumerate(f1_ranking, 1):
            report.append(f"  {i}. {method}: {f1_score:.3f}")
        
        # Speed ranking
        speed_ranking = [(method, stats['methods'][method].get('avg_execution_time', float('inf'))) 
                        for method in self.methods 
                        if stats['methods'][method]['total_analyzed'] > 0 and stats['methods'][method].get('avg_execution_time')]
        speed_ranking.sort(key=lambda x: x[1])
        
        if speed_ranking:
            report.append("\nBy Speed (fastest first):")
            for i, (method, avg_time) in enumerate(speed_ranking, 1):
                report.append(f"  {i}. {method}: {avg_time:.3f}s")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _save_json_report(self, filename, stats):
        output_data = {
            'statistics': stats,
            'raw_results': self.results,
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'methods_tested': self.methods,
                'dataset_paths': self.dataset_paths
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
    
    def _save_csv_report(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            header = ['File', 'GroundTruth'] + [f'{m}_Prediction' for m in self.methods] + \
                    [f'{m}_Confidence' for m in self.methods] + [f'{m}_Time' for m in self.methods] + \
                    ['Ensemble_Prediction', 'Ensemble_Confidence']
            writer.writerow(header)
            
            for result in self.results:
                row = [result['file_name'], result['ground_truth']]

                for method in self.methods:
                    method_result = result['method_results'].get(method, {})
                    if 'error' not in method_result:
                        prediction = 'stego' if method_result.get('is_stego', False) else 'clean'
                        confidence = method_result.get('confidence', 0)
                        exec_time = method_result.get('execution_time', 0)
                    else:
                        prediction = 'error'
                        confidence = 0
                        exec_time = 0
                    
                    row.extend([prediction, confidence, exec_time])
                
                if result.get('ensemble_result'):
                    ensemble = result['ensemble_result']
                    ensemble_pred = ensemble.get('ensemble_classification', 'unknown')
                    ensemble_conf = ensemble.get('ensemble_confidence', 0)
                else:
                    ensemble_pred = 'unavailable'
                    ensemble_conf = 0
                
                row.extend([ensemble_pred, ensemble_conf])
                writer.writerow(row)
    
    def _save_text_report(self, filename, report):
        with open(filename, 'w') as f:
            f.write(report)
    
    def get_method_comparison_table(self):
        if not self.results:
            return []
        
        stats = self._calculate_statistics()
        comparison_data = []
        
        for method in self.methods:
            method_stats = stats['methods'][method]
            if method_stats['total_analyzed'] > 0:
                comparison_data.append({
                    'method': method,
                    'accuracy': f"{method_stats['accuracy']:.3f}",
                    'precision': f"{method_stats['precision']:.3f}",
                    'recall': f"{method_stats['recall']:.3f}",
                    'f1_score': f"{method_stats['f1_score']:.3f}",
                    'avg_time': f"{method_stats.get('avg_execution_time', 0):.3f}s",
                    'errors': method_stats['errors']
                })
        
        ensemble_stats = stats['ensemble']
        if ensemble_stats['total_analyzed'] > 0:
            comparison_data.append({
                'method': 'Ensemble',
                'accuracy': f"{ensemble_stats['accuracy']:.3f}",
                'precision': f"{ensemble_stats['precision']:.3f}",
                'recall': f"{ensemble_stats['recall']:.3f}",
                'f1_score': f"{ensemble_stats['f1_score']:.3f}",
                'avg_time': 'N/A',
                'errors': 0
            })
        
        return comparison_data
    
    def cleanup(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.cleanup()

if __name__ == "__main__":
    # Example with separate cover/stego directories
    dataset_paths = {
        'cover': 'C:\\Users\\lmathope\\Desktop\\Dev\\hyp\\data\\datasets\\cover',
        'stego': 'C:\\Users\\lmathope\\Desktop\\Dev\\hyp\\data\\datasets\\stego'
    }
    
    benchmark = BenchmarkTool(dataset_paths)

    benchmark.run_all_methods_on_dataset(sample_size=2)
    
    report = benchmark.generate_comparison_report('benchmark_results.txt')
    print(report)
    
    benchmark.generate_comparison_report('benchmark_results.json')
    benchmark.generate_comparison_report('benchmark_results.csv')
    
    comparison_table = benchmark.get_method_comparison_table()
    for row in comparison_table:
        print(f"{row['method']}: Accuracy={row['accuracy']}, F1={row['f1_score']}")
    
    benchmark.cleanup()