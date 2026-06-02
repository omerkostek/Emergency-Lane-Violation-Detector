# 
# Central configuration file.
# All constants and tuneable parameters live here so that a
# single edit propagates to every module that imports them.
# 

import os

# Model paths
# Custom-trained YOLO segmentation model for lane detection.
LANE_MODEL_PATH = "train_results/weights/best.pt"

# Pre-trained YOLO model used for vehicle detection & tracking.
VEHICLE_MODEL_PATH = "yolo26n.pt"

# YOLO class filter
# COCO class IDs to detect: 2=car, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 5, 7]

# Lane class IDs (from the custom lane model)
CLASS_NORMAL_LANE = 0
CLASS_UNAUTHORIZED_LANE = 1

# FastALPR model names
PLATE_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
PLATE_OCR_MODEL = "cct-xs-v2-global-model"

# Violation detection thresholds
# A plate is considered "locked" (no more ALPR attempts) once
# its OCR confidence exceeds this value.
PLATE_LOCKED_THRESHOLD = 0.95

# Number of seconds a vehicle must remain in an unauthorized
# lane before a violation is recorded in the database.
VIOLATION_SECONDS_THRESHOLD = 3.0

# Database & I/O paths
DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "Violations_template.db")
INPUT_DIR = "input_videos"
