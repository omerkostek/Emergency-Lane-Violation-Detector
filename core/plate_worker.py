# 
# Contains:
#   - parse_turkish_plate(): regex-based Turkish plate formatter
#   - PlateRecognitionWorker: daemon thread that reads vehicle
#     crops from a queue and runs FastALPR on them.
#
# The worker runs in a separate thread because ALPR inference is
# slow and must not block the main video processing loop.
# The best plate per track_id is kept in the shared `best_plates`
# dict (higher confidence overwrites lower confidence results).
# 

import re
import time
import threading

from fast_alpr import ALPR

from core.config import (
    PLATE_DETECTOR_MODEL,
    PLATE_OCR_MODEL,
    PLATE_LOCKED_THRESHOLD,
)

def parse_turkish_plate(text):
    """Parse and reformat a raw OCR string into Turkish plate format.
    
    Turkish plates follow the pattern: NN AAA NNNN (e.g. '34 ABC 1234').
    Returns the formatted string if it matches, otherwise None.
    """
    text = text.replace(" ", "").upper()
    match = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return None

class PlateRecognitionWorker(threading.Thread):
    """Background daemon thread for license plate recognition.
    
    Consumes (track_id, vehicle_crop) tuples from crop_queue.
    Sends a sentinel (None, None) to the queue to stop the thread.
    Updates best_plates dict in-place (shared with the pipeline loop).
    """

    def __init__(self, crop_queue, best_plates, use_turkish_logic=True):
        super().__init__()
        self.crop_queue = crop_queue
        self.best_plates = best_plates
        self.use_turkish_logic = use_turkish_logic
        self.daemon = True   # Dies automatically when main thread exits
        self.alpr = None

    def run(self):
        # Initialize FastALPR lazily inside the thread (heavy model load)
        print("Initializing FastALPR in background thread...")
        self.alpr = ALPR(
            detector_model=PLATE_DETECTOR_MODEL,
            ocr_model=PLATE_OCR_MODEL,
        )
        print("FastALPR background thread ready.")

        while True:
            try:
                track_id, vehicle_crop = self.crop_queue.get()

                # Sentinel value signals graceful shutdown
                if track_id is None:
                    break

                results = self.alpr.predict(vehicle_crop)

                for res in results:
                    text = res.ocr.text
                    if not text or len(text.strip()) == 0:
                        continue

                    # Apply Turkish plate regex filter if enabled
                    if self.use_turkish_logic:
                        formatted_text = parse_turkish_plate(text)
                        if not formatted_text:
                            continue   # Discard plates that don't match the format
                        text = formatted_text

                    # Average per-character confidences into one score
                    raw_confidence = res.ocr.confidence
                    if isinstance(raw_confidence, list) and len(raw_confidence) > 0:
                        overall_confidence = sum(raw_confidence) / len(raw_confidence)
                    elif isinstance(raw_confidence, (float, int)):
                        overall_confidence = float(raw_confidence)
                    else:
                        overall_confidence = 0.0

                    # Compute normalized bounding box within the crop
                    crop_h, crop_w = vehicle_crop.shape[:2]
                    bbox = res.detection.bounding_box
                    norm_box = (
                        max(0, bbox.x1) / crop_w, max(0, bbox.y1) / crop_h,
                        min(crop_w, bbox.x2) / crop_w, min(crop_h, bbox.y2) / crop_h
                    ) if crop_w > 0 and crop_h > 0 else (0, 0, 0, 0)

                    # Keep only the highest-confidence result per vehicle
                    current_best = self.best_plates.get(track_id, {'confidence': 0.0})
                    if overall_confidence > current_best['confidence']:
                        self.best_plates[track_id] = {
                            'text': text,
                            'confidence': overall_confidence,
                            'time_updated': time.time(),
                            'norm_box': norm_box
                        }

                self.crop_queue.task_done()

            except Exception as e:
                print(f"Error in ALPR worker: {e}")
