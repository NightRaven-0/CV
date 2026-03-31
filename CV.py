import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
import torch
import numpy as np
import os
from datetime import datetime

# Links to YOLOv5 models
# yolov5s: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
# yolov5m: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
# yolov5l: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
# yolov5x: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt

model = None
running = False
results_log = []
alert_object = None
custom_input = None

def load_model(model_name):
    global model
    try:
        model = torch.hub.load("ultralytics/yolov5", model_name)
        messagebox.showinfo("Info", f"Model '{model_name}' loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")

def process_custom_input():
    global custom_input
    file_path = filedialog.askopenfilename(
        filetypes=[("Image and Video Files", "*.jpg *.jpeg *.png *.mp4")]
    )
    if file_path:
        custom_input = file_path
        messagebox.showinfo("Info", f"Selected file: {file_path}")

def start_detection():
    global running, results_log, alert_object, custom_input

    running = True
    results_log = []  # Clear previous results

    cap = (
        cv2.VideoCapture(0)
        if custom_input is None
        else cv2.VideoCapture(custom_input)
    )

    previous_result = None

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Extract detected objects' names and counts
        labels = results.names
        detected_objects = results.pred[0][:, -1].cpu().numpy()
        unique_classes, counts = np.unique(detected_objects, return_counts=True)

        current_result = {labels[int(cls)]: count for cls, count in zip(unique_classes, counts)}

        # If alert object is selected, notify user when detected
        if alert_object and alert_object in current_result:
            messagebox.showwarning(
                "Alert", f"'{alert_object}' detected: {current_result[alert_object]} time(s)"
            )

        # Log results only if they differ from the previous detection
        if current_result != previous_result:
            timestamp = datetime.now().strftime("%H:%M:%S")
            results_log.append(f"[{timestamp}] {current_result}")
            previous_result = current_result

        result_img = np.squeeze(results.render())
        cv2.imshow("YOLOv5 Real-time Object Detection", result_img)

        # Press 'q' to quit the OpenCV window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_detection():
    global running
    running = False
    messagebox.showinfo("Info", "Detection Stopped")

# Save results function
def save_results():
    if results_log:
        save_path = os.path.join(os.getcwd(), f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(save_path, "w") as file:
            file.write("\n".join(results_log))
        messagebox.showinfo("Info", f"Results saved to: {save_path}")
    else:
        messagebox.showinfo("Info", "No results to save.")

def run_detection_in_thread():
    detection_thread = threading.Thread(target=start_detection)
    detection_thread.start()

window = tk.Tk()
window.title("Real-time Object Detection")
window.geometry("400x400")

# Model selection buttons
model_label = tk.Label(window, text="Select YOLOv5 Model:")
model_label.pack(pady=5)

model_buttons = [
    ("YOLOv5s", "yolov5s"),
    ("YOLOv5m", "yolov5m"),
    ("YOLOv5l", "yolov5l"),
    ("YOLOv5x", "yolov5x"),
]
for label, model_name in model_buttons:
    tk.Button(window, text=label, command=lambda m=model_name: load_model(m)).pack(pady=2)

# Custom input button
custom_input_button = tk.Button(window, text="Select Custom Input", command=process_custom_input)
custom_input_button.pack(pady=10)

# Alert dropdown menu
alert_label = tk.Label(window, text="Set Alert for Object:")
alert_label.pack(pady=5)

alert_dropdown = ttk.Combobox(window, state="readonly", values=[])
alert_dropdown.pack(pady=5)

def update_dropdown():
    if model:
        alert_dropdown["values"] = list(model.names.values())
        alert_dropdown.set("")  # Clear previous selection

tk.Button(window, text="Update Alert Options", command=update_dropdown).pack(pady=5)

def set_alert_object():
    global alert_object
    alert_object = alert_dropdown.get()
    messagebox.showinfo("Info", f"Alert set for: {alert_object}")


tk.Button(window, text="Set Alert", command=set_alert_object).pack(pady=5)

# Control buttons
start_button = tk.Button(window, text="Start Detection", command=run_detection_in_thread)
start_button.pack(pady=10)

stop_button = tk.Button(window, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

save_button = tk.Button(window, text="Save Results", command=save_results)
save_button.pack(pady=10)

exit_button = tk.Button(window, text="Exit", command=window.quit)
exit_button.pack(pady=10)

window.mainloop()