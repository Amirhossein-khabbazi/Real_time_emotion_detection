import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

class ObjectDetectionUI:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.root.title("Emotion Detection")
        self.root.geometry("400x400")

        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('TLabel', font=('Helvetica', 14))
        style.configure('TFrame', background='#f0f0f0')

        # Main frame
        self.main_frame = ttk.Frame(root, padding="20 20 20 20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Project title
        self.title_label = ttk.Label(self.main_frame, text="Emotion Detection", font=('Helvetica', 16, 'bold'))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # name
        self.name_label = ttk.Label(self.main_frame, text="Amirhossein Kargar Khabbazi", font=('Helvetica', 14))
        self.name_label.grid(row=1, column=0, columnspan=2, pady=5)

        # GitHub
        self.github_label = ttk.Label(self.main_frame, text="GitHub: Amirhossein-khabbazi", font=('Helvetica', 14))
        self.github_label.grid(row=2, column=0, columnspan=2, pady=5)

        # Start Detection button
        self.start_btn = ttk.Button(self.main_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.grid(row=3, column=0, pady=10, sticky=tk.E)

        # Stop Detection button
        self.stop_btn = ttk.Button(self.main_frame, text="Stop Detection", command=self.stop_detection, state='disabled')
        self.stop_btn.grid(row=3, column=1, pady=10, sticky=tk.W)

        # Capture Photo button
        self.capture_btn = ttk.Button(self.main_frame, text="Capture Photo", command=self.capture_photo, state='disabled')
        self.capture_btn.grid(row=4, column=0, pady=10, sticky=tk.E)

        # Quit button
        self.quit_btn = ttk.Button(self.main_frame, text="Quit", command=self.quit_app)
        self.quit_btn.grid(row=4, column=1, pady=10, sticky=tk.W)

    def start_detection(self):
        self.detector.start_detection()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.capture_btn.config(state='normal')

    def stop_detection(self):
        self.detector.stop_detection()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')

    def capture_photo(self):
        photo_path = self.detector.capture_photo()
        if photo_path:
            messagebox.showinfo("Info", f'Photo captured and saved as {photo_path}')

    def quit_app(self):
        self.detector.stop_detection()
        self.root.quit()
