import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline2 import SteganalysisPipeline
from benchmark.benchmark import BenchmarkTool

class SteganographyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize pipeline and benchmark
        self.pipeline = SteganalysisPipeline()
        self.benchmark_tool = None
        
        # Current analysis results
        self.current_results = {}
        self.current_image_path = None
        
        # Create main interface
        self.create_menu()
        self.create_main_interface()
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.create_status_bar()
        
    def create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Batch Analysis", command=self.batch_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run LSB Analysis", command=lambda: self.run_single_analysis('LSB'))
        analysis_menu.add_command(label="Run DCT Analysis", command=lambda: self.run_single_analysis('DCT'))
        analysis_menu.add_command(label="Run CNN Analysis", command=lambda: self.run_single_analysis('CNN'))
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Run All Methods", command=self.run_all_analyses)
        analysis_menu.add_command(label="Compare Methods", command=self.compare_methods)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Benchmark Models", command=self.benchmark_models)
        tools_menu.add_command(label="Model Performance", command=self.show_model_performance)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_interface(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(notebook)
        notebook.add(self.analysis_frame, text="Analysis")
        self.create_analysis_panel()
        
        # Results tab
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="Results")
        self.create_results_panel()
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(notebook)
        notebook.add(self.comparison_frame, text="Method Comparison")
        self.create_comparison_panel()
        
        # Benchmark tab
        self.benchmark_frame = ttk.Frame(notebook)
        notebook.add(self.benchmark_frame, text="Benchmark")
        self.create_benchmark_panel()
        
    def create_analysis_panel(self):
        # Image selection section
        image_frame = ttk.LabelFrame(self.analysis_frame, text="Image Selection", padding=10)
        image_frame.pack(fill=tk.X, pady=5)
        
        self.image_path_var = tk.StringVar()
        ttk.Label(image_frame, text="Selected Image:").pack(anchor=tk.W)
        image_path_frame = ttk.Frame(image_frame)
        image_path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(image_path_frame, textvariable=self.image_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(image_path_frame, text="Browse", command=self.open_image).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Analysis methods section
        methods_frame = ttk.LabelFrame(self.analysis_frame, text="Detection Methods", padding=10)
        methods_frame.pack(fill=tk.X, pady=5)
        
        # Method selection
        self.method_vars = {}
        methods_grid = ttk.Frame(methods_frame)
        methods_grid.pack(fill=tk.X)
        
        for i, method in enumerate(['LSB', 'DCT', 'CNN']):
            var = tk.BooleanVar(value=True)
            self.method_vars[method] = var
            ttk.Checkbutton(methods_grid, text=f"{method} Analysis", variable=var).grid(row=0, column=i, sticky=tk.W, padx=10)
        
        # Analysis options
        options_frame = ttk.Frame(methods_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        self.quick_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Quick Mode (Faster but less detailed)", variable=self.quick_mode_var).pack(anchor=tk.W)
        
        # Analysis buttons
        buttons_frame = ttk.Frame(methods_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Run Selected Methods", command=self.run_selected_analyses).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Run All Methods", command=self.run_all_analyses).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.analysis_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready to analyze")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
    def create_results_panel(self):
        # Results summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Analysis Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=5)
        
        # Summary grid
        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack(fill=tk.X)
        
        # Summary labels
        labels = ["Overall Result:", "Confidence:", "Methods Agreed:", "Processing Time:"]
        self.summary_vars = {}
        
        for i, label in enumerate(labels):
            ttk.Label(summary_grid, text=label).grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value="N/A")
            self.summary_vars[label] = var
            ttk.Label(summary_grid, textvariable=var, foreground='blue').grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)
        
        # Individual method results
        methods_results_frame = ttk.LabelFrame(self.results_frame, text="Individual Method Results", padding=10)
        methods_results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for method results
        columns = ('Method', 'Result', 'Confidence', 'Time', 'Details')
        self.results_tree = ttk.Treeview(methods_results_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == 'Method':
                self.results_tree.column(col, width=80)
            elif col == 'Result':
                self.results_tree.column(col, width=100)
            elif col == 'Confidence':
                self.results_tree.column(col, width=80)
            elif col == 'Time':
                self.results_tree.column(col, width=80)
            else:
                self.results_tree.column(col, width=200)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(methods_results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        # Detailed results text area
        details_frame = ttk.LabelFrame(self.results_frame, text="Detailed Analysis", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.details_text = scrolledtext.ScrolledText(details_frame, height=10, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
    def create_comparison_panel(self):
        # Comparison controls
        controls_frame = ttk.LabelFrame(self.comparison_frame, text="Comparison Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Compare Current Results", command=self.compare_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Comparison", command=self.export_comparison).pack(side=tk.LEFT, padx=5)
        
        # Comparison table
        table_frame = ttk.LabelFrame(self.comparison_frame, text="Method Performance Comparison", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create comparison treeview
        comp_columns = ('Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Avg Time', 'Errors')
        self.comparison_tree = ttk.Treeview(table_frame, columns=comp_columns, show='headings', height=15)
        
        for col in comp_columns:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, width=100)
        
        self.comparison_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for comparison
        comp_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.comparison_tree.yview)
        comp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comparison_tree.configure(yscrollcommand=comp_scrollbar.set)
        
    def create_benchmark_panel(self):
        # Benchmark controls
        controls_frame = ttk.LabelFrame(self.benchmark_frame, text="Benchmark Configuration", padding=10)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Dataset selection
        dataset_frame = ttk.Frame(controls_frame)
        dataset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dataset_frame, text="Cover Images:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.cover_path_var = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.cover_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_cover_dir).grid(row=0, column=2, padx=5)
        
        ttk.Label(dataset_frame, text="Stego Images:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.stego_path_var = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.stego_path_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_stego_dir).grid(row=1, column=2, padx=5)
        
        # Benchmark options
        options_frame = ttk.Frame(controls_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Sample Size:").pack(side=tk.LEFT, padx=5)
        self.sample_size_var = tk.StringVar(value="10")
        ttk.Entry(options_frame, textvariable=self.sample_size_var, width=10).pack(side=tk.LEFT, padx=5)
        
        self.benchmark_quick_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Quick Mode", variable=self.benchmark_quick_var).pack(side=tk.LEFT, padx=20)
        
        # Benchmark buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Run Benchmark", command=self.run_benchmark).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Report", command=self.save_benchmark_report).pack(side=tk.LEFT, padx=5)
        
        # Benchmark results
        results_frame = ttk.LabelFrame(self.benchmark_frame, text="Benchmark Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.benchmark_text = scrolledtext.ScrolledText(results_frame, height=20, wrap=tk.WORD)
        self.benchmark_text.pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # Add time label
        self.time_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=5)
        self.update_time()
        
    def update_time(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(current_time)
        self.root.after(1000, self.update_time)
        
    def open_image(self):
        filetypes = [
            ("Image files", "*.bmp *.jpg *.jpeg *.png *.tiff *.tif"),
            ("BMP files", "*.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("TIFF files", "*.tiff *.tif"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image for Analysis",
            filetypes=filetypes
        )
        
        if filename:
            self.image_path_var.set(filename)
            self.current_image_path = filename
            self.status_var.set(f"Image loaded: {os.path.basename(filename)}")
            
    def run_single_analysis(self, method):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        for m in self.method_vars:
            self.method_vars[m].set(m == method)
        
        self.run_selected_analyses()
        
    def run_selected_analyses(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        selected_methods = [method for method, var in self.method_vars.items() if var.get()]
        
        if not selected_methods:
            messagebox.showwarning("No Methods", "Please select at least one analysis method.")
            return
        
        # Run analysis in background thread
        thread = threading.Thread(target=self._run_analysis_thread, args=(selected_methods,))
        thread.daemon = True
        thread.start()
        
    def run_all_analyses(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        # Set all methods to true
        for var in self.method_vars.values():
            var.set(True)
        
        self.run_selected_analyses()
        
    def _run_analysis_thread(self, methods):
        try:
            # Update UI
            self.root.after(0, self._start_analysis_ui)
            
            quick_mode = self.quick_mode_var.get()
            results = {}
            
            for i, method in enumerate(methods):
                # Update progress
                self.root.after(0, lambda m=method: self.progress_var.set(f"Running {m} analysis..."))
                
                try:
                    result = self.pipeline.run_analysis(method, self.current_image_path, quick_mode)
                    results[method] = result
                except Exception as e:
                    results[method] = {
                        'method': method,
                        'error': str(e),
                        'is_stego': False,
                        'confidence': 0,
                        'execution_time': 0
                    }
            
            # Generate comparison if multiple methods
            if len(methods) > 1:
                comparison = self.pipeline.compare_results(results)
                results['comparison'] = comparison
            
            self.current_results = results
            
            # Update UI with results
            self.root.after(0, self._complete_analysis_ui)
            
        except Exception as e:
            self.root.after(0, lambda: self._analysis_error_ui(str(e)))
            
    def _start_analysis_ui(self):
        self.progress_bar.start()
        self.status_var.set("Analysis in progress...")
        
    def _complete_analysis_ui(self):
        self.progress_bar.stop()
        self.progress_var.set("Analysis complete")
        self.status_var.set("Analysis completed successfully")
        self.update_results_display()
        
    def _analysis_error_ui(self, error_msg):
        self.progress_bar.stop()
        self.progress_var.set("Analysis failed")
        self.status_var.set("Analysis failed")
        messagebox.showerror("Analysis Error", f"Analysis failed: {error_msg}")
    
    def update_results_display(self):
        """Update the results display with current analysis results."""
        if not self.current_results:
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Update summary
        if 'comparison' in self.current_results:
            comparison = self.current_results['comparison']
            ensemble = comparison['ensemble_prediction']
            consensus = comparison['consensus_analysis']
            
            self.summary_vars["Overall Result:"].set(ensemble['ensemble_classification'].title())
            self.summary_vars["Confidence:"].set(f"{ensemble['ensemble_confidence']:.3f}")
            self.summary_vars["Methods Agreed:"].set(f"{consensus['stego_votes']}/{consensus['total_votes']}")
            
            total_time = sum(r.get('execution_time', 0) for r in self.current_results.values() if isinstance(r, dict) and 'execution_time' in r)
            self.summary_vars["Processing Time:"].set(f"{total_time:.3f}s")
        else:
            # Single method result - Fix for proper classification display
            method_result = None
            
            # Find the first valid method result (skip comparison)
            for method_name, result in self.current_results.items():
                if method_name != 'comparison' and isinstance(result, dict):
                    method_result = result
                    break
            
            if method_result and 'error' not in method_result:
                # Extract classification with proper fallback logic
                classification = method_result.get('classification', 'unknown')
                confidence = method_result.get('confidence', 0)
                is_stego = method_result.get('is_stego', False)
                
                # Fix: Better classification resolution
                if classification == 'unknown' or not classification or classification == '':
                    # Use is_stego as fallback
                    if is_stego:
                        classification = 'stego'
                    else:
                        classification = 'clean'
                
                # Ensure classification is properly formatted
                if classification.lower() in ['true', '1', 'positive']:
                    classification = 'stego'
                elif classification.lower() in ['false', '0', 'negative']:
                    classification = 'clean'
                
                self.summary_vars["Overall Result:"].set(classification.title())
                self.summary_vars["Confidence:"].set(f"{confidence:.3f}")
                self.summary_vars["Methods Agreed:"].set("1/1")
                self.summary_vars["Processing Time:"].set(f"{method_result.get('execution_time', 0):.3f}s")
            else:
                # Error case or no valid results
                error_msg = "Error"
                if method_result and 'error' in method_result:
                    error_msg = "Analysis Error"
                
                self.summary_vars["Overall Result:"].set(error_msg)
                self.summary_vars["Confidence:"].set("0.000")
                self.summary_vars["Methods Agreed:"].set("0/1")
                self.summary_vars["Processing Time:"].set("0.000s")
        
        # Update individual method results
        for method_name, result in self.current_results.items():
            if method_name == 'comparison':
                continue
                
            if 'error' in result:
                self.results_tree.insert('', 'end', values=(
                    method_name,
                    'Error',
                    '0.000',
                    '0.000s',
                    result['error']
                ))
            else:
                # Fix: Consistent classification extraction for individual results
                classification = result.get('classification', 'unknown')
                confidence = result.get('confidence', 0)
                exec_time = result.get('execution_time', 0)
                is_stego = result.get('is_stego', False)
                
                # Apply same classification resolution logic
                if classification == 'unknown' or not classification or classification == '':
                    classification = 'stego' if is_stego else 'clean'
                
                # Handle various classification formats
                if classification.lower() in ['true', '1', 'positive']:
                    classification = 'stego'
                elif classification.lower() in ['false', '0', 'negative']:
                    classification = 'clean'
                
                # Build details string with method-specific information
                details_parts = [f"Method: {method_name}"]
                
                if 'converted_from' in result:
                    details_parts.append(f"Converted from: {result['converted_from']}")
                
                # Add method-specific details
                if method_name == 'LSB':
                    if 'entropy' in result:
                        details_parts.append(f"Entropy: {result['entropy']:.4f}")
                    if 'chi_square' in result:
                        details_parts.append(f"Chi-square: {result['chi_square']:.4f}")
                elif method_name == 'DCT':
                    if 'anomaly_score' in result:
                        details_parts.append(f"Anomaly Score: {result['anomaly_score']:.4f}")
                    if 'dct_entropy' in result:
                        details_parts.append(f"DCT Entropy: {result['dct_entropy']:.4f}")
                elif method_name == 'CNN':
                    details_parts.append("CNN-based prediction")
                    if 'prediction_probabilities' in result:
                        probs = result['prediction_probabilities']
                        if isinstance(probs, dict):
                            prob_str = ", ".join([f"{k}: {v:.3f}" for k, v in probs.items()])
                            details_parts.append(f"Probabilities: {prob_str}")
                
                details = ", ".join(details_parts)
                
                self.results_tree.insert('', 'end', values=(
                    method_name,
                    classification.title(),
                    f"{confidence:.3f}",
                    f"{exec_time:.3f}s",
                    details
                ))
        
        # Update detailed text
        self.update_detailed_text()
    
    def update_detailed_text(self):
        """Update the detailed results text area."""
        self.details_text.delete(1.0, tk.END)
        
        if not self.current_results:
            self.details_text.insert(tk.END, "No analysis results available.")
            return
        
        text_content = []
        text_content.append("=== Detailed Analysis Results ===\n")
        text_content.append(f"Image: {os.path.basename(self.current_image_path) if self.current_image_path else 'Unknown'}\n")
        text_content.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Individual method details
        for method_name, result in self.current_results.items():
            if method_name == 'comparison':
                continue
                
            text_content.append(f"--- {method_name} Analysis ---\n")
            
            if 'error' in result:
                text_content.append(f"Status: ERROR\n")
                text_content.append(f"Error: {result['error']}\n\n")
            else:
                # Fix: Proper classification display
                classification = result.get('classification', 'unknown')
                confidence = result.get('confidence', 0)
                is_stego = result.get('is_stego', False)
                exec_time = result.get('execution_time', 0)
                
                # Fallback if classification is still unknown
                if classification == 'unknown' or not classification:
                    classification = 'stego' if is_stego else 'clean'
                
                text_content.append(f"Classification: {classification.title()}\n")
                text_content.append(f"Is Steganographic: {is_stego}\n")
                text_content.append(f"Confidence: {confidence:.4f}\n")
                text_content.append(f"Execution Time: {exec_time:.4f}s\n")
                
                if 'converted_from' in result:
                    text_content.append(f"Original Format: {result['converted_from']}\n")
                
                # Method-specific details
                if method_name == 'LSB':
                    if 'entropy' in result:
                        text_content.append(f"Entropy: {result['entropy']:.4f}\n")
                    if 'chi_square' in result:
                        text_content.append(f"Chi-square: {result['chi_square']:.4f}\n")
                    if 'z_score' in result:
                        text_content.append(f"Z-score: {result['z_score']:.4f}\n")
                        
                elif method_name == 'DCT':
                    if 'anomaly_score' in result:
                        text_content.append(f"Anomaly Score: {result['anomaly_score']:.4f}\n")
                    if 'dct_entropy' in result:
                        text_content.append(f"DCT Entropy: {result['dct_entropy']:.4f}\n")
                        
                elif method_name == 'CNN':
                    text_content.append("CNN-based deep learning analysis\n")
                    if 'prediction_probabilities' in result:
                        probs = result['prediction_probabilities']
                        if isinstance(probs, dict):
                            text_content.append("Prediction Probabilities:\n")
                            for class_name, prob in probs.items():
                                text_content.append(f"  {class_name}: {prob:.4f}\n")
                    
                    if 'feature_analysis' in result:
                        text_content.append("Feature analysis available\n")
                
                text_content.append("\n")
        
        # Ensemble results
        if 'comparison' in self.current_results:
            comparison = self.current_results['comparison']
            ensemble = comparison['ensemble_prediction']
            consensus = comparison['consensus_analysis']
            
            text_content.append("--- Ensemble Analysis ---\n")
            text_content.append(f"Ensemble Classification: {ensemble['ensemble_classification'].title()}\n")
            text_content.append(f"Ensemble Confidence: {ensemble['ensemble_confidence']:.4f}\n")
            text_content.append(f"Ensemble Score: {ensemble['ensemble_score']:.4f}\n")
            text_content.append(f"Agreement Score: {consensus['agreement_score']:.4f}\n")
            text_content.append(f"Unanimous Decision: {consensus['unanimous']}\n")
            text_content.append(f"Stego Votes: {consensus['stego_votes']}/{consensus['total_votes']}\n")
            text_content.append(f"Contributing Methods: {', '.join(ensemble['contributing_methods'])}\n")
        
        # Insert all text at once
        self.details_text.insert(tk.END, ''.join(text_content))    
    
    def compare_current_results(self):
        if not self.current_results:
            messagebox.showinfo("No Results", "No analysis results to compare. Please run an analysis first.")
            return
        
        # Clear comparison tree
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        
        # If we have comparison data, show it
        if 'comparison' in self.current_results:
            comparison = self.current_results['comparison']
            
            # Individual method performance
            for method_name, result in self.current_results.items():
                if method_name == 'comparison':
                    continue
                
                if 'error' not in result:
                    confidence = result.get('confidence', 0)
                    exec_time = result.get('execution_time', 0)
                    
                    # still under development
                    self.comparison_tree.insert('', 'end', values=(
                        method_name,
                        f"{confidence:.3f}",
                        "N/A",
                        "N/A",
                        "N/A", 
                        f"{exec_time:.3f}s",
                        "0"
                    ))
            
            # Add ensemble row
            ensemble = comparison['ensemble_prediction']
            self.comparison_tree.insert('', 'end', values=(
                "Ensemble",
                f"{ensemble['ensemble_confidence']:.3f}",
                "N/A",
                "N/A", 
                "N/A",
                "N/A",
                "0"
            ))
        else:
            messagebox.showinfo("Single Method", "Comparison requires multiple methods. Run analysis with multiple methods selected.")
    
    def compare_methods(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        

        for var in self.method_vars.values():
            var.set(True)
        
        self.run_selected_analyses()
        
    def batch_analysis(self):
        directory = filedialog.askdirectory(title="Select Directory for Batch Analysis")
        if not directory:
            return
    
        output_file = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        # Run batch analysis in background
        thread = threading.Thread(target=self._run_batch_thread, args=(directory, output_file))
        thread.daemon = True
        thread.start()
        
    def _run_batch_thread(self, directory, output_file):
        try:
            self.root.after(0, lambda: self.status_var.set("Running batch analysis..."))
            self.root.after(0, self.progress_bar.start)
            
            # Get selected methods
            selected_methods = [method for method, var in self.method_vars.items() if var.get()]
            if not selected_methods:
                selected_methods = list(self.pipeline.methods.keys())
            
            quick_mode = self.quick_mode_var.get()
            
            results = self.pipeline.batch_analyze(directory, selected_methods, quick_mode, output_file)
            
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.status_var.set("Batch analysis completed"))
            self.root.after(0, lambda: messagebox.showinfo("Batch Complete", f"Batch analysis completed. Results saved to {output_file}"))
            
        except Exception as e:
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.status_var.set("Batch analysis failed"))
            self.root.after(0, lambda: messagebox.showerror("Batch Error", f"Batch analysis failed: {str(e)}"))
    
    def browse_cover_dir(self):
        directory = filedialog.askdirectory(title="Select Cover Images Directory")
        if directory:
            self.cover_path_var.set(directory)
    
    def browse_stego_dir(self):
        directory = filedialog.askdirectory(title="Select Stego Images Directory")
        if directory:
            self.stego_path_var.set(directory)
    
    def run_benchmark(self):
        cover_dir = self.cover_path_var.get()
        stego_dir = self.stego_path_var.get()
        
        if not cover_dir or not stego_dir:
            messagebox.showwarning("Missing Directories", "Please select both cover and stego image directories.")
            return
        
        try:
            sample_size = int(self.sample_size_var.get())
        except ValueError:
            sample_size = 10
        
        # Run benchmark in background
        thread = threading.Thread(target=self._run_benchmark_thread, args=(cover_dir, stego_dir, sample_size))
        thread.daemon = True
        thread.start()
        
    def _run_benchmark_thread(self, cover_dir, stego_dir, sample_size):
        try:
            self.root.after(0, lambda: self.status_var.set("Running benchmark..."))
            self.root.after(0, self.progress_bar.start)
            
            # Create benchmark tool
            dataset_paths = {
                'cover': cover_dir,
                'stego': stego_dir
            }
            
            self.benchmark_tool = BenchmarkTool(dataset_paths)
            
            # Run benchmark
            quick_mode = self.benchmark_quick_var.get()
            self.benchmark_tool.run_all_methods_on_dataset(quick_mode=quick_mode, sample_size=sample_size)
            
            # Generate report
            report = self.benchmark_tool.generate_comparison_report()
            
            # Update UI
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self._update_benchmark_results(report))
            self.root.after(0, lambda: self.status_var.set("Benchmark completed"))
            
        except Exception as e:
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.status_var.set("Benchmark failed"))
            self.root.after(0, lambda: messagebox.showerror("Benchmark Error", f"Benchmark failed: {str(e)}"))
    
    def _update_benchmark_results(self, report):
        self.benchmark_text.delete(1.0, tk.END)
        self.benchmark_text.insert(tk.END, report)
        
        # Update comparison table if benchmark tool available
        if self.benchmark_tool:
            comparison_data = self.benchmark_tool.get_method_comparison_table()
            
            # Clear comparison tree
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            
            # Add benchmark data to comparison
            for row in comparison_data:
                self.comparison_tree.insert('', 'end', values=(
                    row['method'],
                    row['accuracy'],
                    row['precision'],
                    row['recall'],
                    row['f1_score'],
                    row['avg_time'],
                    row['errors']
                ))
    
    def save_benchmark_report(self):
        if not self.benchmark_tool:
            messagebox.showwarning("No Benchmark", "No benchmark results to save. Please run benchmark first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Benchmark Report",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                self.benchmark_tool.generate_comparison_report(filename)
                messagebox.showinfo("Report Saved", f"Benchmark report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save report: {str(e)}")
    
    def benchmark_models(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first to run model comparison.")
            return

        self.run_all_analyses()
        
    def show_model_performance(self):
        if self.benchmark_tool:
            comparison_data = self.benchmark_tool.get_method_comparison_table()
            
            performance_text = "=== Model Performance Summary ===\n\n"
            
            for row in comparison_data:
                performance_text += f"Method: {row['method']}\n"
                performance_text += f"  Accuracy: {row['accuracy']}\n"
                performance_text += f"  Precision: {row['precision']}\n"
                performance_text += f"  Recall: {row['recall']}\n"
                performance_text += f"  F1-Score: {row['f1_score']}\n"
                performance_text += f"  Avg Time: {row['avg_time']}\n"
                performance_text += f"  Errors: {row['errors']}\n\n"
            
            # Show in a dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Model Performance")
            dialog.geometry("400x500")
            
            text_widget = scrolledtext.ScrolledText(dialog, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, performance_text)
            text_widget.config(state=tk.DISABLED)
        else:
            messagebox.showinfo("No Data", "No performance data available. Please run benchmark first.")
    
    def export_comparison(self):
        filename = filedialog.asksaveasfilename(
            title="Export Comparison Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Avg Time', 'Errors'])
                    
                    # Write data from comparison tree
                    for item in self.comparison_tree.get_children():
                        values = self.comparison_tree.item(item)['values']
                        writer.writerow(values)
                
                messagebox.showinfo("Export Complete", f"Comparison results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def clear_results(self):
        self.current_results = {}
        
        # Clear summary
        for var in self.summary_vars.values():
            var.set("N/A")
        
        # Clear results tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Clear details text
        self.details_text.delete(1.0, tk.END)
        
        self.status_var.set("Results cleared")
    
    def show_about(self):
        about_text = """
Steganography Detection System

This application provides comprehensive steganography detection using multiple analysis methods:

• LSB Analysis - Detects LSB-based steganography
• DCT Analysis - Frequency domain analysis for JPEG steganography  
• CNN Analysis - Deep learning-based detection

Features:
• Multi-method analysis with ensemble prediction
• Batch processing capabilities
• Performance benchmarking
• Method comparison and evaluation
        """
        
        messagebox.showinfo("About", about_text.strip())

def main():
    root = tk.Tk()
    app = SteganographyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()