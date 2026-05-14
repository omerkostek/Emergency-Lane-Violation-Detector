# Emergency Lane Violation Detector

A real-time AI system that detects vehicles illegally using emergency lanes and logs violations with license plate recognition.

## About

The system uses a custom-trained YOLO segmentation model to identify normal and emergency lanes, then tracks vehicles with ByteTrack to detect when they enter restricted lanes. A violation is recorded only after a vehicle remains in an unauthorized lane for 3+ continuous seconds. All violations are stored in a SQLite database and viewable through a live web dashboard.

## Features

- Automatic emergency lane detection via AI segmentation
- Real-time multi-vehicle tracking (cars, buses, trucks)
- Turkish license plate recognition with 95% confidence locking
- 3-second violation threshold (mirrors real EDS camera behavior)
- Live web dashboard with video stream and violation list
- SQLite logging with timestamp, plate text, and OCR confidence

## Tech Stack

- **Ultralytics YOLO** — vehicle detection & lane segmentation
- **FastALPR** — license plate recognition (ONNX backend)
- **FastAPI + Uvicorn** — web dashboard
- **OpenCV** — video processing
- **SQLite** — violation database

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python dashboard.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

1. Select a video file from the dropdown
2. Click **Start** — the system will detect lanes automatically
3. Confirm the detected lanes, then live tracking begins
4. Violations are logged in real time to the database
