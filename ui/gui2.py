import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import json
from datetime import datetime

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        # Will be initialized when imported
        self.pipeline = None  
        self.file_path = None
        self.current_image = None
        self.analysis_results = {}
        self.analysis_history = []
        self.processing_thread = None
        self.is_processing = False
        
        # UI State variables
        self.selected_method = tk.StringVar(value="CNN Deep Learning")
        self.batch_mode = tk.BooleanVar(value=False)
        self.save_results = tk.BooleanVar(value=True)
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.auto_preprocess = tk.BooleanVar(value=True)
        
        # Results storage
        self.results_data = {
            'probability': 0.0,
            'classification': 'Unknown',
            'confidence': 0.0,
            'processing_time': 0.0,
            'method_used': '',
            'file_path': '',
            'timestamp': ''
        }
        
        self.init_ui()
        self.setup_menu()
        self.center_window()
        
    def init_ui(self):
        #Initialize GUI elements
        self.root.title(" Steganalysis Accuracy Detection ")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles
        self.setup_styles()
        
        #Main frame with padding
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Steganalysis Accuracy Detection",
            font=('Arial', 20, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        self.create_control_panel(main_frame)
        self.create_image_display_area(main_frame)
        self.create_results_panel(main_frame)
        self.create_status_bar(main_frame)
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Action.TButton', 
                       background='#3498db',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(10, 5))
        
        style.configure('Danger.TButton',
                       background='#e74c3c',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(10, 5))
        
        style.configure('Success.TButton',
                       background='#27ae60',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(10, 5))
        
    def setup_menu(self):
        #application menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.select_file)
        file_menu.add_command(label="Open Folder", command=self.select_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results", command=self.save_results_to_file)
        file_menu.add_command(label="Export Report", command=self.export_detailed_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Quick Scan", command=self.quick_analysis)
        analysis_menu.add_command(label="Deep Analysis", command=self.deep_analysis)
        analysis_menu.add_command(label="Batch Processing", command=self.toggle_batch_mode)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="View History", command=self.show_history_window)
        analysis_menu.add_command(label="Clear History", command=self.clear_analysis_history)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Model Comparison", command=self.show_model_comparison)
        tools_menu.add_command(label="Benchmark Performance", command=self.benchmark_models)
        tools_menu.add_command(label="Settings", command=self.show_settings_window)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        
    def create_control_panel(self, parent):
        control_frame = tk.LabelFrame(
            parent,
            text="Analysis Controls",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection section
        file_section = tk.Frame(control_frame, bg='#f0f0f0')
        file_section.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            file_section,
            text="Image File:",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT)
        
        self.file_label = tk.Label(
            file_section,
            text="No file selected",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#7f8c8d',
            width=50,
            anchor='w'
        )
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            file_section,
            text="Browse",
            command=self.select_file,
            style='Action.TButton'
        ).pack(side=tk.RIGHT)
        
        # Method selection section
        method_section = tk.Frame(control_frame, bg='#f0f0f0')
        method_section.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            method_section,
            text="Detection Method:",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT)
        
        methods = [
            "CNN Deep Learning",
            "Statistical Analysis", 
            "Frequency Domain",
        ]
        
        method_combo = ttk.Combobox(
            method_section,
            textvariable=self.selected_method,
            values=methods,
            state='readonly',
            width=20
        )
        method_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Options section
        options_section = tk.Frame(control_frame, bg='#f0f0f0')
        options_section.pack(fill=tk.X, pady=(0, 15))
        
        # Checkboxes for options
        tk.Checkbutton(
            options_section,
            text="Auto Preprocessing",
            variable=self.auto_preprocess,
            bg='#f0f0f0',
            font=('Arial', 9)
        ).pack(side=tk.LEFT)
        
        tk.Checkbutton(
            options_section,
            text="Save Results",
            variable=self.save_results,
            bg='#f0f0f0',
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Checkbutton(
            options_section,
            text="Batch Mode",
            variable=self.batch_mode,
            bg='#f0f0f0',
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(20, 0))
        
        # Threshold section
        threshold_section = tk.Frame(control_frame, bg='#f0f0f0')
        threshold_section.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            threshold_section,
            text="Confidence Threshold:",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT)
        
        self.threshold_scale = tk.Scale(
            threshold_section,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.confidence_threshold,
            bg='#f0f0f0',
            length=200
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=(10, 0))
        
        self.threshold_label = tk.Label(
            threshold_section,
            text="0.50",
            font=('Arial', 10),
            bg='#f0f0f0',
            width=5
        )
        self.threshold_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Update threshold label when scale changes
        self.threshold_scale.config(command=self.update_threshold_label)
        
        # Action buttons section
        button_section = tk.Frame(control_frame, bg='#f0f0f0')
        button_section.pack(fill=tk.X)
        
        self.analyze_button = ttk.Button(
            button_section,
            text="Analyze Image",
            command=self.run_selected_method,
            style='Success.TButton'
        )
        self.analyze_button.pack(side=tk.LEFT)
        
        self.stop_button = ttk.Button(
            button_section,
            text="Stop Analysis",
            command=self.stop_analysis,
            style='Danger.TButton',
            state='disabled'
        )
        self.stop_button.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            button_section,
            text="Clear Results",
            command=self.clear_results,
            style='Action.TButton'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            button_section,
            text="Compare Methods",
            command=self.compare_all_methods,
            style='Action.TButton'
        ).pack(side=tk.RIGHT)
        
    def create_image_display_area(self, parent):
        display_frame = tk.LabelFrame(
            parent,
            text="Image Preview",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))        
        self.image_canvas = tk.Canvas(
            display_frame,
            bg='white',
            relief=tk.SUNKEN,
            bd=2
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.image_info_label = tk.Label(
            display_frame,
            text="No image loaded",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.image_info_label.pack(pady=(5, 0))
        
    def create_results_panel(self, parent):
        results_frame = tk.LabelFrame(
            parent,
            text="Analysis Results",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        results_frame.config(width=350)
        
        # Detection result section
        detection_frame = tk.Frame(results_frame, bg='#f0f0f0')
        detection_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            detection_frame,
            text="Detection Result:",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        self.result_label = tk.Label(
            detection_frame,
            text="No analysis performed",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#7f8c8d',
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=10
        )
        self.result_label.pack(fill=tk.X, pady=(5, 0))
        
        # Confidence section
        confidence_frame = tk.Frame(results_frame, bg='#f0f0f0')
        confidence_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            confidence_frame,
            text="Confidence Score:",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="0.00%",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        self.confidence_label.pack(anchor='w', pady=(5, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            confidence_frame,
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))        
        metrics_frame = tk.Frame(results_frame, bg='#f0f0f0')
        metrics_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            metrics_frame,
            text="Detailed Metrics:",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        # Create scrollable text widget for metrics
        metrics_text_frame = tk.Frame(metrics_frame)
        metrics_text_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.metrics_text = tk.Text(
            metrics_text_frame,
            height=8,
            width=40,
            font=('Courier', 9),
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            bd=1
        )
        
        metrics_scrollbar = tk.Scrollbar(metrics_text_frame)
        metrics_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.metrics_text.config(yscrollcommand=metrics_scrollbar.set)
        metrics_scrollbar.config(command=self.metrics_text.yview)
        
        # Processing time section
        time_frame = tk.Frame(results_frame, bg='#f0f0f0')
        time_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            time_frame,
            text="Processing Time:",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        self.time_label = tk.Label(
            time_frame,
            text="0.00 seconds",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        self.time_label.pack(anchor='w', pady=(5, 0))
        result_buttons_frame = tk.Frame(results_frame, bg='#f0f0f0')
        result_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(
            result_buttons_frame,
            text="Save Results",
            command=self.save_current_results,
            style='Action.TButton'
        ).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(
            result_buttons_frame,
            text="View Details",
            command=self.show_detailed_results,
            style='Action.TButton'
        ).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(
            result_buttons_frame,
            text="Export Report",
            command=self.export_current_report,
            style='Action.TButton'
        ).pack(fill=tk.X)
        
    def create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg='#34495e', relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=2)
        self.activity_label = tk.Label(
            status_frame,
            text="●",
            font=('Arial', 12),
            bg='#34495e',
            fg='#27ae60'
        )
        self.activity_label.pack(side=tk.RIGHT, padx=10, pady=2)
        
    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def select_file(self):
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.gif'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Image for Analysis",
            filetypes=filetypes
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path), fg='#2c3e50')
            self.load_and_display_image(file_path)
            self.update_status(f"Image loaded: {os.path.basename(file_path)}")
            
    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder for Batch Analysis")
        
        if folder_path:
            self.batch_mode.set(True)
            self.file_path = folder_path
            self.file_label.config(text=f"Folder: {os.path.basename(folder_path)}", fg='#2c3e50')
            self.update_status(f"Folder selected for batch processing: {os.path.basename(folder_path)}")
            
    def load_and_display_image(self, file_path):
        try:
            # Open image with PIL
            pil_image = Image.open(file_path)
            self.current_image = pil_image.copy()
            
            # Calculate display size while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 400, 300
            
            img_width, img_height = pil_image.size
            
            # Calculate scaling factor
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image for display
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(display_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")

            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            self.image_canvas.create_image(x, y, anchor='nw', image=self.photo_image)
            
            file_size = os.path.getsize(file_path) / 1024  # KB
            self.image_info_label.config(
                text=f"Size: {img_width}x{img_height} | File: {file_size:.1f} KB",
                fg='#2c3e50'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.update_status("Error loading image")
            
    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{float(value):.2f}")
        
    def run_selected_method(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return
            
        if self.is_processing:
            messagebox.showinfo("Info", "Analysis is already in progress.")
            return
        self.processing_thread = threading.Thread(target=self._perform_analysis)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _perform_analysis(self):
        try:
            self.is_processing = True
            self.root.after(0, self._update_ui_processing_start)
            
            # Simulate analysis process
            method = self.selected_method.get()
            start_time = time.time()
            
            # Update status
            self.root.after(0, lambda: self.update_status(f"Analyzing with {method}..."))
            
            # Simulation of different processing times and results based on method
            if method == "CNN Deep Learning":
                processing_time = 2.5
                probability = np.random.uniform(0.65, 0.95)
                classification = "Steganographic" if probability > self.confidence_threshold.get() else "Clean"
            elif method == "Statistical Analysis":
                processing_time = 1.2
                probability = np.random.uniform(0.45, 0.85)
                classification = "Steganographic" if probability > self.confidence_threshold.get() else "Clean"
            elif method == "Frequency Domain":
                processing_time = 1.8
                probability = np.random.uniform(0.55, 0.88)
                classification = "Steganographic" if probability > self.confidence_threshold.get() else "Clean"           
            # Simulate processing with progress updates
            steps = 20
            for i in range(steps + 1):
                progress = (i / steps) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                time.sleep(processing_time / steps)
                
                if not self.is_processing:
                    return
            
            end_time = time.time()
            actual_processing_time = end_time - start_time
            
            # Prepare results
            results = {
                'probability': probability,
                'classification': classification,
                'confidence': probability * 100,
                'processing_time': actual_processing_time,
                'method_used': method,
                'file_path': self.file_path,
                'timestamp': datetime.now().isoformat(),
                'threshold_used': self.confidence_threshold.get(),
                'auto_preprocess': self.auto_preprocess.get()
            }
            
            self.analysis_history.append(results.copy())
            self.root.after(0, lambda: self._update_ui_with_results(results))
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_analysis_error(str(e)))
        finally:
            self.is_processing = False
            self.root.after(0, self._update_ui_processing_end)
            
    def _update_ui_processing_start(self):
        self.analyze_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.activity_label.config(fg='#f39c12')
        
    def _update_ui_processing_end(self):
        self.analyze_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.activity_label.config(fg='#27ae60')
        
    def _update_ui_with_results(self, results):
        self.results_data = results
        classification = results['classification']
        if classification == "Steganographic":
            self.result_label.config(
                text="STEGANOGRAPHIC CONTENT DETECTED",
                fg='#e74c3c',
                bg='#fadbd8'
            )
        else:
            self.result_label.config(
                text="CLEAN IMAGE",
                fg='#27ae60',
                bg='#d5f4e6'
            )
        
        # Update confidence
        confidence = results['confidence']
        self.confidence_label.config(text=f"{confidence:.2f}%")
        self.progress_var.set(confidence)
        
        # Update processing time
        self.time_label.config(text=f"{results['processing_time']:.2f} seconds")
        
        # Update detailed metrics
        self.update_detailed_metrics(results)
        
        # Update status
        self.update_status(f"Analysis complete: {classification}")
        
        # Save results if option is enabled
        if self.save_results.get():
            self.save_current_results()
            
    def update_detailed_metrics(self, results):
        self.metrics_text.delete(1.0, tk.END)
        
        metrics_text = f"""Analysis Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method: {results['method_used']}
File: {os.path.basename(results['file_path'])}
Timestamp: {results['timestamp'][:19]}

Detection Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classification: {results['classification']}
Probability: {results['probability']:.4f}
Confidence: {results['confidence']:.2f}%
Threshold: {results['threshold_used']:.2f}

Performance Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Processing Time: {results['processing_time']:.3f}s
Auto Preprocess: {'Yes' if results['auto_preprocess'] else 'No'}

Additional Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analysis ID: {len(self.analysis_history)}
Status: Complete
"""
        
        self.metrics_text.insert(1.0, metrics_text)
        
    def _handle_analysis_error(self, error_message):
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{error_message}")
        self.update_status("Analysis failed")
        self.result_label.config(
            text="ANALYSIS FAILED",
            fg='#e74c3c',
            bg='#fadbd8'
        )
        
    def stop_analysis(self):
        if self.is_processing:
            self.is_processing = False
            self.update_status("Analysis stopped by user")
            messagebox.showinfo("Info", "Analysis has been stopped.")
            
    def clear_results(self):
        self.result_label.config(
            text="No analysis performed",
            fg='#7f8c8d',
            bg='#f0f0f0'
        )
        self.confidence_label.config(text="0.00%")
        self.progress_var.set(0)
        self.time_label.config(text="0.00 seconds")
        self.metrics_text.delete(1.0, tk.END)
        self.update_status("Results cleared")
        
    def quick_analysis(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return            
        self.selected_method.set("Statistical Analysis")
        self.confidence_threshold.set(0.3)
        self.run_selected_method()
        
    def deep_analysis(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return
            
        # Set to most thorough method, 
        self.selected_method.set("Ensemble Methods")
        self.confidence_threshold.set(0.7)
        self.auto_preprocess.set(True)
        self.run_selected_method()
        
    def toggle_batch_mode(self):
        current_state = self.batch_mode.get()
        self.batch_mode.set(not current_state)
        
        if self.batch_mode.get():
            self.select_folder()
        else:
            messagebox.showinfo("Info", "Batch mode disabled. Select individual images.")
            
    def compare_all_methods(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return
            
        # Show comparison window
        self.show_method_comparison_window()
        
    def show_method_comparison_window(self):
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Method Comparison")
        comparison_window.geometry("800x600")
        comparison_window.configure(bg='#f0f0f0')
        
        #comparison interface
        tk.Label(
            comparison_window,
            text="Method Performance Comparison",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        ).pack(pady=20)
        columns = ['Method', 'Classification', 'Confidence', 'Time (s)', 'Status']
        
        tree_frame = tk.Frame(comparison_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        button_frame = tk.Frame(comparison_window, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(
            button_frame,
            text="Run Comparison",
            command=lambda: self.run_method_comparison(tree),
            style='Success.TButton'
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            button_frame,
            text="Export Results",
            command=lambda: self.export_comparison_results(tree),
            style='Action.TButton'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Close",
            command=comparison_window.destroy,
            style='Action.TButton'
        ).pack(side=tk.RIGHT)
        
    def run_method_comparison(self, tree):
        methods = [
            "CNN Deep Learning",
            "Statistical Analysis",
            "Frequency Domain", 
        ]        
        for item in tree.get_children():
            tree.delete(item)
        
        for method in methods:
            start_time = time.time()
            
            # Simulation, I will finalize development different results for each method
            if method == "CNN Deep Learning":
                classification = "Steganographic"
                confidence = 87.5
                proc_time = 2.3
            elif method == "Statistical Analysis":
                classification = "Clean"
                confidence = 72.1
                proc_time = 1.1
            elif method == "Frequency Domain":
                classification = "Steganographic"
                confidence = 79.8
                proc_time = 1.7
            else: 
                classification = "Steganographic"
                confidence = 94.7
                proc_time = 3.8
            
            tree.insert('', 'end', values=[
                method,
                classification,
                f"{confidence:.1f}%",
                f"{proc_time:.1f}",
                "Complete"
            ])            
            tree.update()
            time.sleep(0.5)
            
    def export_comparison_results(self, tree):
        file_path = filedialog.asksaveasfilename(
            title="Save Comparison Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Method Comparison Results\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for item in tree.get_children():
                        values = tree.item(item)['values']
                        f.write(f"Method: {values[0]}\n")
                        f.write(f"Classification: {values[1]}\n")
                        f.write(f"Confidence: {values[2]}\n")
                        f.write(f"Processing Time: {values[3]}\n")
                        f.write(f"Status: {values[4]}\n")
                        f.write("-" * 30 + "\n")
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                
    def save_current_results(self):
        if not self.results_data.get('classification'):
            messagebox.showwarning("Warning", "No results to save.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"steganalysis_result_{timestamp}.json"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".json",
            initialvalue=filename,
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.results_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                self.update_status(f"Results saved: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                
    def save_results_to_file(self):
        if not self.analysis_history:
            messagebox.showwarning("Warning", "No analysis history to save.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"steganalysis_history_{timestamp}.json"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis History",
            defaultextension=".json",
            initialvalue=filename,
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    # Save as CSV
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        if self.analysis_history:
                            writer = csv.DictWriter(f, fieldnames=self.analysis_history[0].keys())
                            writer.writeheader()
                            writer.writerows(self.analysis_history)
                else:
                    # Save as JSON
                    with open(file_path, 'w') as f:
                        json.dump(self.analysis_history, f, indent=2)
                
                messagebox.showinfo("Success", f"History saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save history: {str(e)}")
                
    def show_history_window(self):
        if not self.analysis_history:
            messagebox.showinfo("Info", "No analysis history available.")
            return
            
        history_window = tk.Toplevel(self.root)
        history_window.title("Analysis History")
        history_window.geometry("900x600")
        history_window.configure(bg='#f0f0f0')
        
        # Title
        tk.Label(
            history_window,
            text="Analysis History",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        ).pack(pady=20)
        
        # History table
        columns = ['Timestamp', 'File', 'Method', 'Result', 'Confidence', 'Time']
        
        tree_frame = tk.Frame(history_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate history
        for item in self.analysis_history:
            tree.insert('', 'end', values=[
                item['timestamp'][:19],
                os.path.basename(item['file_path']),
                item['method_used'],
                item['classification'],
                f"{item['confidence']:.1f}%",
                f"{item['processing_time']:.2f}s"
            ])
        
        # Buttons
        button_frame = tk.Frame(history_window, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(
            button_frame,
            text="Clear History",
            command=lambda: self.clear_history_and_refresh(tree),
            style='Danger.TButton'
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            button_frame,
            text="Export History",
            command=self.save_results_to_file,
            style='Action.TButton'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Close",
            command=history_window.destroy,
            style='Action.TButton'
        ).pack(side=tk.RIGHT)
        
    def clear_history_and_refresh(self, tree):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all analysis history?"):
            self.clear_analysis_history()
            for item in tree.get_children():
                tree.delete(item)
                
    def clear_analysis_history(self):
        self.analysis_history.clear()
        self.update_status("Analysis history cleared")
        
    def show_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#f0f0f0')
        
        tk.Label(
            settings_window,
            text="Application Settings",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        ).pack(pady=20)
        
        tk.Label(
            settings_window,
            text="Settings panel will be implemented here",
            font=('Arial', 12),
            bg='#f0f0f0'
        ).pack(pady=20)
        
    def show_user_guide(self):
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        guide_window.configure(bg='#f0f0f0')
        
        # User guide content
        guide_text = """
Steganalysis Detection Tool - User Guide

1. Getting Started:
   • Click "Browse" to select an image file
   • Choose your preferred detection method
   • Adjust confidence threshold if needed
   • Click "Analyze Image" to start

2. Detection Methods:
   • CNN Deep Learning: Most accurate, slower
   • Statistical Analysis: Fast, good for basic detection
   • Frequency Domain: Good for DCT-based steganography

3. Understanding Results:
   • Green = Clean image (no hidden data detected)
   • Red = Steganographic content detected
   • Confidence shows detection certainty

4. Advanced Features:
   • Batch Mode: Process multiple images
   • Method Comparison: Compare all methods
   • History: View past analysis results
   • Export: Save results and reports

5. Tips:
   • Higher threshold = fewer false positives
   • Use ensemble methods for critical analysis
   • Enable auto-preprocessing for better results
        """
        
        text_widget = tk.Text(
            guide_window,
            wrap=tk.WORD,
            font=('Arial', 10),
            padx=20,
            pady=20
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(1.0, guide_text)
        text_widget.config(state='disabled')
        
    def show_about_dialog(self):
        """Show about dialog."""
        about_text = """
 Steganalysis Detection Tool v2.0

A comprehensive tool for detecting hidden data in digital images using advanced machine learning and statistical analysis techniques.

Features:
• Multiple detection algorithms
• Batch processing
• Detailed reporting
• Method comparison

        """
        
        messagebox.showinfo("About", about_text)
        
    def export_detailed_report(self):
        if not self.results_data.get('classification'):
            messagebox.showwarning("Warning", "No analysis results to export.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"steganalysis_report_{timestamp}.txt"
        
        file_path = filedialog.asksaveasfilename(
            title="Export Detailed Report",
            defaultextension=".txt",
            initialvalue=filename,
            filetypes=[("Text files", "*.txt"), ("HTML files", "*.html")]
        )
        
        if file_path:
            try:
                report = self.generate_detailed_report()
                
                with open(file_path, 'w') as f:
                    f.write(report)
                
                messagebox.showinfo("Success", f"Report exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
                
    def generate_detailed_report(self):
        if not self.results_data:
            return "No analysis data available."
            
        report = f"""
STEGANALYSIS DETECTION REPORT
{'=' * 60}

Analysis Summary:
File: {os.path.basename(self.results_data.get('file_path', 'Unknown'))}
Method: {self.results_data.get('method_used', 'Unknown')}
Timestamp: {self.results_data.get('timestamp', 'Unknown')}

Results:
Classification: {self.results_data.get('classification', 'Unknown')}
Confidence: {self.results_data.get('confidence', 0):.2f}%
Probability: {self.results_data.get('probability', 0):.4f}
Processing Time: {self.results_data.get('processing_time', 0):.3f} seconds

Configuration:
Threshold Used: {self.results_data.get('threshold_used', 0):.2f}
Auto Preprocessing: {self.results_data.get('auto_preprocess', False)}

Analysis Details:
{'=' * 60}
The analysis was performed using {self.results_data.get('method_used', 'Unknown')} method.
The image was classified as {self.results_data.get('classification', 'Unknown')} with a confidence of {self.results_data.get('confidence', 0):.2f}%.

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report
        
    def export_current_report(self):
        self.export_detailed_report()
        
    def show_detailed_results(self):
        if not self.results_data.get('classification'):
            messagebox.showwarning("Warning", "No analysis results to display.")
            return
            
        details_window = tk.Toplevel(self.root)
        details_window.title("Detailed Analysis Results")
        details_window.geometry("700x600")
        details_window.configure(bg='#f0f0f0')
        
        text_frame = tk.Frame(details_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Courier', 10),
            padx=20,
            pady=20
        )
        
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)        
        detailed_report = self.generate_detailed_report()
        text_widget.insert(1.0, detailed_report)
        text_widget.config(state='disabled')
        
        ttk.Button(
            details_window,
            text="Close",
            command=details_window.destroy,
            style='Action.TButton'
        ).pack(pady=10)
        
    def show_model_comparison(self):        
        self.show_method_comparison_window()
        
    def benchmark_models(self):
        #Benchmarking 
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return
            
        messagebox.showinfo("Benchmark", "Model benchmarking will be implemented in future versions.")
        
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def run(self):
        try:
            #pipeline initialisation
            from pipeline.pipeline import SteganalysisPipeline
            self.pipeline = SteganalysisPipeline()
            self.update_status("Pipeline initialized successfully")
        except ImportError:
            self.update_status("Warning: Pipeline module not found - running in demo mode")
            
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    app = MainWindow()
    app.run()