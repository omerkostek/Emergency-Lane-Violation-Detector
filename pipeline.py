# pipeline.py
# ─────────────────────────────────────────────────────────────
# Orchestrator for the full video processing pipeline.
# This module only coordinates the phases; business logic
# lives in the core/ sub-modules.
#
# Pipeline phases:
#   Phase 1+2 — AI lane detection + user confirmation (calibration)
#   Phase 3   — Vehicle tracking + ALPR + violation detection
# ─────────────────────────────────────────────────────────────

import os
import queue
import time
import datetime
from collections import defaultdict

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


# ── Calibration ───────────────────────────────────────────────

def _calibration_web(cap, web_state: WebState):
    """Phase 1+2: detect lanes frame by frame until user confirms.

    Jumps 30 frames at a time when the user clicks 'Next Frame'.
    Returns (normal_polys, unauthorized_polys) or None on abort.
    """
    frame_count = 0
    web_state.state = "detecting"

    while not web_state.confirm_event.is_set():
        if frame_count > 0:
            for _ in range(29):
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


# ── HUD drawing ───────────────────────────────────────────────

def _draw_vehicle_hud(frame, x1, y1, x2, y2, track_id, box_color,
                       status_text, status_color, plate_info,
                       use_turkish_logic, attempt_count):
    """Draw bounding box, plate text, and violation status for one vehicle."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    display_text = f"ID: {track_id}"

    if plate_info:
        display_text += f" | {plate_info['text']} ({plate_info['confidence']:.2f})"
        if 'norm_box' in plate_info and plate_info['confidence'] < PLATE_LOCKED_THRESHOLD:
            nx1, ny1, nx2, ny2 = plate_info['norm_box']
            car_w, car_h = x2 - x1, y2 - y1
            px1, py1 = x1 + int(nx1 * car_w), y1 + int(ny1 * car_h)
            px2, py2 = x1 + int(nx2 * car_w), y1 + int(ny2 * car_h)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
            cv2.putText(frame, plate_info['text'], (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    elif use_turkish_logic and attempt_count >= 6:
        display_text += " | Plate Unreadable!"
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    if status_text:
        cv2.putText(frame, status_text, (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


# ── Main entry point ──────────────────────────────────────────

def process_video_stream(video_source: str, use_turkish_logic: bool,
                         web_state: WebState):
    """Run the full ALPR violation detection pipeline on a video file."""
    print(f"Loading vehicle tracking model ({VEHICLE_MODEL_PATH})...")
    yolo_model = YOLO(VEHICLE_MODEL_PATH)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # ── Phase 1+2: Lane calibration ───────────────────────────
    print("--- PHASE 1: AI LANE DETECTION ---")

    result = _calibration_web(cap, web_state)
    if result is None:
        return

    normal_polys, unauthorized_polys = result

    # ── Phase 3: Vehicle tracking + ALPR + violations ─────────
    print("--- PHASE 3: STARTING LIVE TRACKING ---")

    conn, cur = init_db()

    # Restart video from the beginning so tracking covers the full footage
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    crop_queue = queue.Queue(maxsize=100)
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

    violation_mgr = ViolationManager(video_fps, VIOLATION_SECONDS_THRESHOLD)
    web_state.state = "tracking"

    # ── Main tracking loop ────────────────────────────────────
    while cap.isOpened():
        while web_state.pause_event.is_set() and not web_state.stop_event.is_set():
            time.sleep(0.1)
        if web_state.stop_event.is_set():
            break

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

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = y2

                in_unauthorized = any(
                    cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
                    for poly in unauthorized_polys
                )

                status = violation_mgr.update(track_id, in_unauthorized)

                if (status == "violation"
                        and track_id not in violation_dict
                        and violation_mgr.already_violated(track_id)):
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    vsec = int(frame_count / video_fps)
                    src_name = os.path.basename(video_source)
                    db_id = insert_violation(cur, ts, src_name, vsec)
                    conn.commit()
                    violation_dict[track_id] = {
                        "db_row_id": db_id,
                        "timestamp": ts,
                        "video_source": src_name,
                        "video_second": vsec,
                    }

                if status == "violation":
                    box_color, status_text, status_color = (0, 0, 255), "VIOLATION!", (0, 0, 255)
                elif status == "warning":
                    box_color, status_text, status_color = (0, 165, 255), "WARNING", (0, 165, 255)
                else:
                    box_color, status_text, status_color = (0, 255, 0), None, None

                plate_info = best_plates.get(track_id)
                is_locked = (
                    plate_info is not None
                    and plate_info['confidence'] >= PLATE_LOCKED_THRESHOLD
                )
                if not is_locked and (frame_count - last_queued_frame[track_id] >= 5):
                    last_queued_frame[track_id] = frame_count
                    if not crop_queue.full():
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

        for t_id, v_data in violation_dict.items():
            db_id = v_data.get("db_row_id")
            if db_id is not None:
                p_info = best_plates.get(t_id)
                if p_info:
                    update_plate(cur, db_id, p_info['text'], p_info['confidence'])
        conn.commit()

        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.rectangle(annotated_frame, (10, 10), (260, 80), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        web_state.push_frame(annotated_frame)
        web_state.fps = current_fps
        web_state.frame_count = frame_count
        if web_state.stop_event.is_set():
            print("Web client stopped the stream.")
            break

    # ── Cleanup ───────────────────────────────────────────────
    crop_queue.put((None, None))
    cap.release()

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
