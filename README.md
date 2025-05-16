# PerclosTrakingUAO


This project implements a visual fatigue and drowsiness monitoring system based on the PERCLOS index (Percentage of Eye Closure), using MediaPipe and OpenCV. It detects blinks and signs of drowsiness in real-time through the webcam.

## 🚀 Features

- Face tracking using MediaPipe FaceMesh
- Eye Aspect Ratio (EAR) calculation to detect blinks
- Drowsiness detection based on closed-eye duration
- Automatic brightness and contrast adjustment
- Real-time statistics:
  - Total blink count
  - Blink frequency (per minute)
  - Drowsiness events detected

## 🛠️ Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt

👨‍💻 Authors
Roger Alvarez

Simon Barrera

Arturo Fawcett
