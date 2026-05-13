# core/lane_detector.py
# ─────────────────────────────────────────────────────────────
# Runs the custom YOLO segmentation model on a single frame to
# identify normal and unauthorized lanes.
#
# Key design decisions:
#   - The model is loaded fresh every call during calibration
#     (called only a few times, not per frame in tracking).
#   - Masks are read from results[0].masks.xy (pixel polygons),
#     NOT from bounding boxes, because the model is a segment task.
#   - draw_lane_overlays() is a reusable helper used both in
#     calibration frames and in the live tracking loop.
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import (
    LANE_MODEL_PATH,
    CLASS_NORMAL_LANE,
    CLASS_UNAUTHORIZED_LANE,
)


def detect_lanes_with_model(frame):
    """Run the custom segmentation model on one frame.

    Returns:
        normal_polys (list[np.ndarray]): Polygons for normal lanes.
        unauthorized_polys (list[np.ndarray]): Polygons for unauthorized lanes.
        annotated_frame (np.ndarray): Frame with overlays and legend drawn.
    """
    # Load model and run inference (conf=0.3 keeps borderline detections)
    lane_model = YOLO(LANE_MODEL_PATH)
    results = lane_model.predict(frame, verbose=False, conf=0.3)

    normal_polys = []
    unauthorized_polys = []
    annotated = frame.copy()

    # Parse segmentation masks — each mask is a list of (x, y) pixel coords
    if results and results[0].masks:
        masks = results[0].masks.xy
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for mask_pts, cls_id in zip(masks, classes):
            if len(mask_pts) < 3:
                # Need at least 3 points to form a valid polygon
                continue
            poly = np.array(mask_pts, np.int32)
            if cls_id == CLASS_NORMAL_LANE:
                normal_polys.append(poly)
            elif cls_id == CLASS_UNAUTHORIZED_LANE:
                unauthorized_polys.append(poly)

    # Draw transparent fills and boundary lines on the annotated copy
    annotated = draw_lane_overlays(annotated, normal_polys, unauthorized_polys)

    # Legend box (top-left corner)
    cv2.rectangle(annotated, (10, 10), (300, 75), (0, 0, 0), -1)
    cv2.putText(annotated, f"Normal lanes: {len(normal_polys)}", (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
    cv2.putText(annotated, f"Unauthorized lanes: {len(unauthorized_polys)}", (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return normal_polys, unauthorized_polys, annotated


def draw_lane_overlays(frame, normal_polys, unauthorized_polys, alpha=0.15):
    """Draw transparent colored fills and boundary lines for detected lanes.
    
    Alpha controls fill opacity (0.0 = invisible, 1.0 = solid).
    Normal lanes → green; Unauthorized lanes → red.
    """
    if normal_polys or unauthorized_polys:
        overlay = frame.copy()
        for poly in normal_polys:
            cv2.fillPoly(overlay, [poly], (0, 200, 0))      # Green fill
        for poly in unauthorized_polys:
            cv2.fillPoly(overlay, [poly], (0, 0, 200))      # Red fill
        # Blend the overlay with the original frame for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Solid boundary lines on top of the transparent fill
    for poly in normal_polys:
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 220, 0), thickness=2)
    for poly in unauthorized_polys:
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

    return frame
