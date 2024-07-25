import tkinter as tk
from ui import ObjectDetectionUI
from detector import ObjectDetector

if __name__ == "__main__":
    model_path = './../models/Yolo10L_Emotion_Detection.pt'
    detector = ObjectDetector(model_path)
    
    root = tk.Tk()
    app = ObjectDetectionUI(root, detector)
    root.mainloop()
