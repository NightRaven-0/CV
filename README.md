# Real-Time Object Detection GUI using YOLOv5

A simple desktop application built with **Python**, **Tkinter**, **OpenCV**, and **PyTorch** for running **real-time object detection** with pretrained **YOLOv5** models.

## Overview

This project provides a GUI-driven workflow for object detection. The user can choose a YOLOv5 variant, use a webcam or a custom media file as input, monitor a selected object class through an alert system, and save timestamped detection results to a text file.

The repository is lightweight and focused on inference, not model training. It is best understood as a desktop computer vision prototype or mini-project that demonstrates end-to-end integration between GUI programming and deep learning inference.

## Features

- Select from **YOLOv5s, YOLOv5m, YOLOv5l, and YOLOv5x**
- Use **webcam input** or a **custom image/video file**
- Run **real-time detection** and display annotated frames
- Populate an alert list from model class names
- Trigger a warning when a chosen object is detected
- Save session results as a **timestamped text log**

## Tech Stack

- **Python**
- **Tkinter** for the desktop interface
- **OpenCV** for video capture and frame display
- **PyTorch Hub** for loading YOLOv5 models
- **NumPy** for class counting
- **threading** for running detection without freezing the GUI

## Project Structure

```text
CV/
├── CV.py
└── yolov5s.pt
```

## How It Works

1. Launch the GUI.
2. Load one of the YOLOv5 model variants.
3. Optionally choose a custom image or video file. If no file is chosen, the webcam is used.
4. Update the alert dropdown after the model loads.
5. Choose an object class for alerts.
6. Start detection.
7. View detections in the OpenCV output window.
8. Save the results log if needed.

## Installation

Install the required Python packages:

```bash
pip install opencv-python numpy
```

Install **PyTorch** using the command recommended for your system from the official PyTorch installer, then run the project with:

```bash
python CV.py
```

> Note: Tkinter is included with many Python installations, but on some Linux systems you may need to install it separately.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PyTorch
- Tkinter
- Internet access for PyTorch Hub model download (in the current implementation)

## Known Limitations

- The app assumes a model has been loaded before detection starts.
- The local `yolov5s.pt` file is present in the repository, but the current code loads models through PyTorch Hub instead of using the local file.
- Alert dialogs may appear repeatedly during continuous detection.
- There are no user controls for confidence threshold or IoU threshold.
- The project is written in a single script and does not include `requirements.txt`.

## Future Improvements

- Add support for loading local custom weights
- Add confidence and IoU sliders
- Add better exception handling and startup validation
- Show logs directly inside the GUI
- Save outputs as CSV or JSON
- Modularize the code into GUI, inference, and utilities
- Add a `requirements.txt` and screenshots to the repository

## License

No license file is currently included in the repository. Add one if you plan to distribute or reuse the project publicly.
