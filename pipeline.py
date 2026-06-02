# 
# Orchestrator for the full video processing pipeline.
# This module only coordinates the phases; business logic
# lives in the core/ sub-modules.
#
# Pipeline phases:
#   Phase 1+2 — AI lane detection + user confirmation (calibration)
#   Phase 3   — Vehicle tracking + ALPR + violation detection
# 

import os
import csv
import queue
import time
import datetime
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from core.config import (
    VEHICLE_MODEL_PATH,
    VEHICLE_CLASSES,
    PLATE_LOCKED_THRESHOLD,
    VIOLATION_SECONDS_THRESHOLD,
)
from core.lane_detector import detect_lanes_with_model, draw_lane_overlays
from core.plate_worker import PlateRecognitionWorker
from core.violation_manager import ViolationManager
from core.database import (
    init_db,
    insert_violation,
    update_plate,
    finalize_unresolved,
    get_violation_count,
)
from core.state import WebState

PROFILE_MODE = False  # Set to True to record per-frame timing data to CSV
PROFILE_LOG_PATH         = Path("fps_logs_v2/fps_profile.csv")
PROFILE_WARMUP_SECONDS   = 10   # Frames during this window are skipped (model warm-up)
PROFILE_DURATION_SECONDS = 60   # How long to measure after the warm-up period

# Calibration

def _calibration_web(cap, web_state: WebState):
    """Phase 1+2: detect lanes frame by frame until user confirms.

    Jumps 10 frames at a time when the user clicks 'Next Frame'.
    Returns (normal_polys, unauthorized_polys) or None on abort.
    """
    frame_count = 0
    web_state.state = "detecting"

    while not web_state.confirm_event.is_set():
        if frame_count > 0:
            for _ in range(9):
                cap.read()
                frame_count += 1

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame for calibration.")
            return None

        frame_count += 1
        web_state.frame_height, web_state.frame_width = frame.shape[:2]
        web_state.state = "detecting"
        web_state.push_frame(frame)

        print(f"Running lane detection on frame {frame_count}...")
        normal_polys, unauthorized_polys, cal_frame = detect_lanes_with_model(frame)
        print(f"  → {len(normal_polys)} normal, {len(unauthorized_polys)} unauthorized lane(s)")

        web_state.state = "calibrating"
        web_state.next_frame_event.clear()

        while not web_state.confirm_event.is_set() and not web_state.next_frame_event.is_set():
            if web_state.stop_event.is_set():
                cap.release()
                return None
            web_state.push_frame(cal_frame)
            time.sleep(0.1)

        if web_state.confirm_event.is_set():
            print("User confirmed lanes. Starting tracking...")
            return normal_polys, unauthorized_polys

    return None


# HUD drawing

def _draw_vehicle_hud(frame, x1, y1, x2, y2, track_id, box_color,
                       status_text, status_color, plate_info,
                       use_turkish_logic, attempt_count):
    """Draw bounding box, plate text, and violation status for one vehicle."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Sol üst — Vehicle ID
    cv2.putText(frame, f"ID: {track_id}", (x1 + 4, y1 + 18), font, 0.6, box_color, 2)

    if plate_info:
        # Sol alt — Plate text
        cv2.putText(frame, plate_info['text'], (x1 + 4, y2 - 6), font, 0.6, box_color, 2)

        # Sağ üst — Confidence
        conf_text = f"{plate_info['confidence']:.2f}"
        (conf_w, _), _ = cv2.getTextSize(conf_text, font, 0.6, 2)
        cv2.putText(frame, conf_text, (x2 - conf_w - 4, y1 + 18), font, 0.6, box_color, 2)

    elif use_turkish_logic and attempt_count >= 6:
        cv2.putText(frame, "PLATE?", (x1 + 4, y2 - 6), font, 0.6, (0, 0, 255), 2)

    if status_text:
        cv2.putText(frame, status_text, (x1 + 4, y2 - 22), font, 0.6, status_color, 2)


# Main entry point

def process_video_stream(video_source: str, use_turkish_logic: bool,
                         web_state: WebState):
    """Run the full ALPR violation detection pipeline on a video file."""
    print(f"Loading vehicle tracking model ({VEHICLE_MODEL_PATH})...")
    yolo_model = YOLO(VEHICLE_MODEL_PATH)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Phase 1+2: Lane calibration
    print("--- PHASE 1: AI LANE DETECTION ---")

    result = _calibration_web(cap, web_state)
    if result is None:
        return

    normal_polys, unauthorized_polys = result

    # Phase 3: Vehicle tracking + ALPR + violations
    print("--- PHASE 3: STARTING LIVE TRACKING ---")

    conn, cur = init_db()

    # Restart video from the beginning so tracking covers the full footage
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    # Queue that feeds vehicle crops to the background ALPR thread.
    # maxsize=100 prevents memory buildup if ALPR falls behind.
    crop_queue = queue.Queue(maxsize=100)

    # Shared dict written by the ALPR thread, read by the pipeline thread.
    # Structure — key: track_id (int), value: {text, confidence, time_updated, norm_box}
    best_plates: dict = {}

    alpr_worker = PlateRecognitionWorker(crop_queue, best_plates, use_turkish_logic)
    alpr_worker.start()

    start_time = time.time()
    last_queued_frame: dict = defaultdict(int)
    attempt_counts: dict = defaultdict(int)
    violation_dict: dict = {}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30

    violation_mgr = ViolationManager(VIOLATION_SECONDS_THRESHOLD)

    web_state.state = "tracking"

    if PROFILE_MODE:
        PROFILE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        profile_log = open(PROFILE_LOG_PATH, "w", newline="")
        profile_writer = csv.writer(profile_log)
        profile_writer.writerow(["frame_idx", "wall_time_sec", "t_total_ms", "instantaneous_fps"])
        pipeline_start_time = time.perf_counter()

    # Main tracking loop
    while cap.isOpened():
        while web_state.pause_event.is_set() and not web_state.stop_event.is_set():
            time.sleep(0.1)
        if web_state.stop_event.is_set():
            break

        if PROFILE_MODE:
            t_frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1
        annotated_frame = frame.copy()

        draw_lane_overlays(annotated_frame, normal_polys, unauthorized_polys)

        results = yolo_model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES, verbose=False
        )

        # ByteTrack only assigns IDs when it has tracks; skip frames with no detections.
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Lane violation test
                # Use the bottom-center point as the vehicle's road contact point.
                # cv2.pointPolygonTest returns ≥ 0 if the point is inside the polygon.
                cx_mid = float((x1 + x2) / 2)
                cy_bot = float(y2)
                in_unauthorized = any(
                    cv2.pointPolygonTest(poly, (cx_mid, cy_bot), False) >= 0
                    for poly in unauthorized_polys
                )

                # Violation state machine
                # Returns "safe" / "warning" / "violation" based on elapsed time.
                status = violation_mgr.update(track_id, in_unauthorized)

                # Write violation to database (once per track_id)
                # Insert immediately when the vehicle first crosses the threshold.
                # Plate column starts as "Scanning..." and is updated by the ALPR
                # thread (or via the per-frame update block below).
                if (status == "violation"
                        and track_id not in violation_dict
                        and violation_mgr.already_violated(track_id)):
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    vsec = int(frame_count / video_fps)
                    vtime = f"{vsec // 60}:{vsec % 60:02d}"
                    src_name = os.path.basename(video_source)

                    db_id = insert_violation(cur, ts, src_name, vtime)
                    existing_plate = best_plates.get(track_id)
                    if existing_plate:
                        update_plate(cur, db_id, existing_plate['text'], existing_plate['confidence'])
                    conn.commit()

                    violation_dict[track_id] = {
                        "db_row_id": db_id,
                        "timestamp": ts,
                        "video_source": src_name,
                        "video_second": vsec,
                    }

                # HUD color logic
                if status == "violation":   #RED
                    box_color, status_text, status_color = (0, 0, 255), "VIOLATION!", (0, 0, 255)
                elif status == "warning":   #ORANGE
                    box_color, status_text, status_color = (0, 165, 255), "WARNING!", (0, 165, 255)
                else:   #GREEN
                    box_color, status_text, status_color = (0, 255, 0), None, None

                # ALPR crop queuing
                # Send a vehicle crop to the background ALPR thread every 10 frames.
                # Stop sending once the plate is locked (confidence above threshold).
                plate_info = best_plates.get(track_id)
                is_locked = (
                    plate_info is not None
                    and plate_info['confidence'] >= PLATE_LOCKED_THRESHOLD
                )
                if not is_locked and (frame_count - last_queued_frame[track_id] >= 10):
                    last_queued_frame[track_id] = frame_count
                    if not crop_queue.full():
                        # Clamp coordinates to frame boundaries before cropping
                        y1_c = max(0, y1); y2_c = min(frame.shape[0], y2)
                        x1_c = max(0, x1); x2_c = min(frame.shape[1], x2)
                        crop = frame[y1_c:y2_c, x1_c:x2_c].copy()
                        if crop.size > 0:
                            crop_queue.put_nowait((track_id, crop))
                            attempt_counts[track_id] += 1

                _draw_vehicle_hud(
                    annotated_frame, x1, y1, x2, y2,
                    track_id, box_color, status_text, status_color,
                    plate_info, use_turkish_logic, attempt_counts[track_id]
                )


        # Every frame: refresh plate text for all recorded violators.
        # The ALPR thread may have found a higher-confidence result since
        # the violation was first written, so we overwrite with the latest best.
        for t_id, v_data in violation_dict.items():
            db_id = v_data.get("db_row_id")
            if db_id is not None:
                p_info = best_plates.get(t_id)
                if p_info:
                    update_plate(cur, db_id, p_info['text'], p_info['confidence'])
        conn.commit()

        # FPS overlay + web frame push
        # Average FPS since pipeline start; displayed in the top-left corner.
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.rectangle(annotated_frame, (10, 10), (260, 80), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode to JPEG and put in the shared buffer for the MJPEG endpoint.
        web_state.push_frame(annotated_frame)
        web_state.fps = current_fps
        web_state.frame_count = frame_count

        # Per-frame profiling
        # Measures total wall time per frame (ms) and instantaneous FPS.
        # Results are flushed immediately so partial CSVs are readable on crash.
        if PROFILE_MODE:
            wall_time = time.perf_counter() - pipeline_start_time
            if wall_time >= PROFILE_WARMUP_SECONDS:
                t_total = (time.perf_counter() - t_frame_start) * 1000
                inst_fps = 1000.0 / t_total if t_total > 0 else 0
                profile_writer.writerow([
                    frame_count, f"{wall_time:.3f}",
                    f"{t_total:.2f}", f"{inst_fps:.2f}"
                ])
                profile_log.flush()
            if wall_time >= PROFILE_WARMUP_SECONDS + PROFILE_DURATION_SECONDS:
                print(f"Profile duration reached ({PROFILE_DURATION_SECONDS}s). Stopping.")
                break

        if web_state.stop_event.is_set():
            print("Web client stopped the stream.")
            break

    # Cleanup
    if PROFILE_MODE:
        profile_log.close()
        print(f"Profile log saved to: {PROFILE_LOG_PATH}")

    crop_queue.put((None, None))
    cap.release()

    # Finalize any unresolved violations in the database by marking them as "Ghost Vehicle"
    for t_id, v_data in violation_dict.items():
        db_id = v_data.get("db_row_id")
        if db_id is not None:
            finalize_unresolved(cur, db_id)
    conn.commit()

    vid_name = os.path.basename(video_source)
    total = get_violation_count(cur, vid_name)
    print(f"\nTotal violations recorded for {vid_name}: {total}")

    try:
        conn.close()
    except Exception:
        pass

    web_state.state = "done"
