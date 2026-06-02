# Emergency Lane Violation Detector

A real-time AI system that detects vehicles illegally using emergency lanes and logs violations with license plate recognition.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Project Structure](#2-project-structure)
3. [Execution Flow — Step by Step](#3-execution-flow--step-by-step)
4. [Module Explanations](#4-module-explanations)
5. [Technologies Used and Why](#5-technologies-used-and-why)
6. [Critical Design Decisions](#6-critical-design-decisions)
7. [Database Schema](#7-database-schema)
8. [Configuration Parameters](#8-configuration-parameters)
9. [Installation and Usage](#9-installation-and-usage)

---

## 1. What It Does

This system processes traffic camera footage and automatically detects vehicles that illegally enter emergency (shoulder) lanes. It mirrors the behavior of real EDS (Electronic Detection System) enforcement cameras used in traffic law enforcement.

**Core pipeline:**
1. A custom-trained YOLO segmentation model identifies normal and emergency lanes in the video frame.
2. The operator confirms the detected lanes via a web dashboard.
3. Vehicle tracking begins. Each vehicle is tracked continuously across frames using ByteTrack.
4. Every frame, the system checks whether each vehicle's position overlaps with an unauthorized lane polygon.
5. A vehicle must remain in the unauthorized lane for **3 continuous seconds** before a violation is recorded — momentary crossings are ignored.
6. When a violation is confirmed, a row is written to a SQLite database. In parallel, a license plate recognition thread reads the vehicle's crop and writes the plate text and confidence back to that row.

---

## 2. Project Structure

```
Emergency Lane Violation Detector/
│
├── dashboard.py                  # Entry point — starts the FastAPI web server
├── pipeline.py                   # Orchestrator — runs all 3 pipeline phases
│
├── core/
│   ├── config.py                 # All tunable constants (thresholds, paths, model names)
│   ├── state.py                  # WebState — shared memory between pipeline thread and web server
│   ├── lane_detector.py          # YOLO segmentation — detects lane polygons
│   ├── plate_worker.py           # Background thread — runs FastALPR on vehicle crops
│   ├── violation_manager.py      # State machine — tracks violation timer per vehicle
│   └── database.py               # All SQLite operations (insert, update, finalize)
│
├── api/
│   ├── routes_pipeline.py        # /api/start, /api/stop, /api/pause, /api/resume
│   ├── routes_calibration.py     # /api/confirm, /api/next_frame
│   └── routes_data.py            # /api/status, /api/videos, /api/violations, /video_feed
│
├── templates/
│   └── index.html                # Single-page web dashboard (served by FastAPI)
│
├── input_videos/                 # Video files to process (.mp4)
├── database/                     # SQLite database files
└── train_results/                # Custom-trained lane segmentation model weights
```

---

## 3. Execution Flow — Step by Step

This section follows the exact order in which code runs when you start the system.

### Step 1 — Launch the server (`dashboard.py`)

```bash
python dashboard.py
```

`dashboard.py` creates a FastAPI application and mounts three routers:

| Router file | Prefix | Purpose |
|---|---|---|
| `api/routes_pipeline.py` | `/api` | Pipeline lifecycle control |
| `api/routes_calibration.py` | `/api` | Lane calibration control |
| `api/routes_data.py` | `/api`, `/video_feed` | Read-only data and MJPEG stream |

Uvicorn starts listening on `http://127.0.0.1:8000`. The browser loads `templates/index.html`.

---

### Step 2 — User selects a video and clicks Start (`api/routes_pipeline.py`)

The dashboard sends:
```
POST /api/start  {"video": "video1.mp4"}
```

`start_pipeline()` in `routes_pipeline.py`:
1. Stops any currently running pipeline (sets `stop_event`, joins the old thread).
2. Creates a fresh `WebState` object.
3. Imports `process_video_stream` from `pipeline.py` (lazy import to avoid circular dependency).
4. Launches `process_video_stream()` in a new **daemon thread**.
5. Returns `{"status": "started"}` immediately — the pipeline runs in the background.

---

### Step 3 — Phase 1+2: Lane Calibration (`pipeline.py` → `core/lane_detector.py`)

Inside `process_video_stream()`:

```
YOLO model loaded → VideoCapture opened → _calibration_web() called
```

`_calibration_web()` runs a loop:
- Reads one frame from the video.
- Calls `detect_lanes_with_model(frame)` from `core/lane_detector.py`.
- `detect_lanes_with_model()` loads the custom segmentation model (`train_results/weights/best.pt`) and runs `model.predict()`.
- The model returns **pixel-level polygon masks** (not bounding boxes — it's a segmentation task).
- Masks are split by class: `CLASS_NORMAL_LANE (0)` → green polygons, `CLASS_UNAUTHORIZED_LANE (1)` → red polygons.
- The annotated frame is pushed to `web_state.frame_buffer` so the browser can see it via the MJPEG stream.
- The loop waits until the user either clicks **Confirm** or **Next Frame**.

**Confirm** → browser sends `POST /api/confirm` → `routes_calibration.py` sets `web_state.confirm_event` → calibration loop exits → Phase 3 begins.

**Next Frame** → browser sends `POST /api/next_frame` → loop skips 10 frames and re-runs lane detection.

---

### Step 4 — Phase 3: Main Tracking Loop (`pipeline.py`)

After calibration, `process_video_stream()` resets the video to frame 0 and enters the main loop:

```python
while cap.isOpened():
    ...
```

**Every iteration (one frame):**

#### 4a. YOLO vehicle detection + ByteTrack

```python
results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml",
                            classes=VEHICLE_CLASSES)
```

- `VEHICLE_CLASSES = [2, 5, 7]` — COCO IDs for car, bus, truck.
- `persist=True` tells YOLO to carry ByteTrack state across frames.
- ByteTrack assigns a persistent `track_id` to each vehicle. The ID stays the same across frames even when the vehicle is partially occluded.
- Returns bounding boxes (`xyxy`) and `track_id` for each detected vehicle.

#### 4b. Unauthorized lane check

For each tracked vehicle:
```python
cx_mid = float((x1 + x2) / 2)
cy_bot = float(y2)
in_unauthorized = any(
    cv2.pointPolygonTest(poly, (cx_mid, cy_bot), False) >= 0
    for poly in unauthorized_polys
)
```

- The bottom-center point of the bounding box is used — this corresponds to the contact point between the vehicle and the road surface.
- `cv2.pointPolygonTest` checks if that point is inside any unauthorized lane polygon.
- If the point is inside → the vehicle is considered to be in an unauthorized lane.

#### 4d. Violation state machine (`core/violation_manager.py`)

`ViolationManager.update(track_id, in_unauthorized)` returns one of three states:

| State | Condition | Result |
|---|---|---|
| `"safe"` | Vehicle is in a normal lane | Timer reset |
| `"warning"` | Vehicle entered unauthorized lane, timer running | Orange bounding box |
| `"violation"` | Vehicle stayed ≥ 3 seconds in unauthorized lane | Red bounding box, DB write |

The timer is wall-clock time (`time.time()`), not frame count — so it works correctly at any video FPS.

Once a vehicle is flagged as `"violation"`, it stays in that state for the rest of the session (the `_violated` set persists).

#### 4e. Database write (`core/database.py`)

When a vehicle first transitions to `"violation"` (and is not already in `violation_dict`):

1. `insert_violation(cur, timestamp, video_name, video_time)` — inserts a row with `license_plate = "Scanning..."`.
2. If `best_plates` already has a plate for this vehicle (from earlier ALPR attempts), `update_plate()` is called immediately.
3. The `db_row_id` is stored in `violation_dict[track_id]`.

Every frame, for all vehicles in `violation_dict`:
```python
update_plate(cur, db_id, plate_text, confidence)
conn.commit()
```
This continuously overwrites the plate field as ALPR confidence improves.

#### 4f. ALPR crop queue (`core/plate_worker.py`)

Every 5 frames per vehicle (if not yet "locked"):
```python
crop_queue.put_nowait((track_id, vehicle_crop))
```

- The crop is a pixel cutout of the vehicle's bounding box from the current frame.
- "Locked" means the plate confidence already reached `PLATE_LOCKED_THRESHOLD = 0.95` — no more attempts needed.

#### 4g. Dashboard frame push

```python
web_state.push_frame(annotated_frame)
```

`push_frame()` in `core/state.py` JPEG-encodes the annotated frame (quality 80) and stores it in `frame_buffer` behind a `threading.Lock`. The MJPEG stream endpoint in `routes_data.py` reads from this buffer at ~30 fps.

---

### Step 5 — Background: ALPR Thread (`core/plate_worker.py`)

`PlateRecognitionWorker` runs as a daemon thread started alongside the main loop:

```
crop_queue (shared) → ALPR.predict(crop) → best_plates[track_id] updated
```

For each crop dequeued:
1. `FastALPR.predict()` runs plate detection (YOLO-based) + OCR on the crop.
2. The OCR result is a text string + a list of per-character confidence floats.
3. Per-character confidences are averaged into a single `overall_confidence`.
4. If `use_turkish_logic=True`, `parse_turkish_plate()` validates the text against the Turkish plate regex (`^\d{2}[A-Z]{1,3}\d{2,4}$`) and reformats it. Non-matching results are discarded.
5. If `overall_confidence > best_plates[track_id]['confidence']`, the dict entry is overwritten.

The word "daemon" means this thread is automatically killed when the main thread exits — no cleanup needed.

---

### Step 6 — Cleanup

When the video ends or the user clicks Stop:
- `crop_queue.put((None, None))` — sentinel value tells the ALPR thread to exit cleanly.
- `finalize_unresolved()` — any rows still showing `"Scanning..."` are updated to `"Unreadable"`.
- `profile_log.close()` — FPS profiling CSV is flushed and closed (if `PROFILE_MODE = True`).
- `web_state.state = "done"` — dashboard shows the final state.

---

## 4. Module Explanations

### `dashboard.py` — Entry Point

Creates the FastAPI app, mounts all three routers, serves `index.html` at `/`. Starts Uvicorn on port 8000. Everything else is triggered by HTTP requests from the browser.

---

### `pipeline.py` — Orchestrator

The single largest file. Contains:
- `_calibration_web()` — Phase 1+2 loop
- `_draw_vehicle_hud()` — draws bounding box, plate text, confidence, status label on frame
- `process_video_stream()` — the main Phase 3 loop

Also contains the FPS profiling constants (`PROFILE_MODE`, `PROFILE_LOG_PATH`, `PROFILE_WARMUP_SECONDS`, `PROFILE_DURATION_SECONDS`) which write per-frame timing data to a CSV for performance analysis.

---

### `core/config.py` — Central Configuration

All constants are defined here so a single edit propagates everywhere:

| Constant | Value | Purpose |
|---|---|---|
| `LANE_MODEL_PATH` | `train_results/weights/best.pt` | Custom segmentation model |
| `VEHICLE_MODEL_PATH` | `yolo26n.pt` | Pre-trained vehicle detection model |
| `VEHICLE_CLASSES` | `[2, 5, 7]` | COCO IDs: car, bus, truck |
| `CLASS_NORMAL_LANE` | `0` | Lane model class ID |
| `CLASS_UNAUTHORIZED_LANE` | `1` | Lane model class ID |
| `PLATE_DETECTOR_MODEL` | `yolo-v9-t-384-...` | FastALPR plate detector |
| `PLATE_OCR_MODEL` | `cct-xs-v2-global-model` | FastALPR OCR model |
| `PLATE_LOCKED_THRESHOLD` | `0.95` | Stop sending ALPR crops above this confidence |
| `VIOLATION_SECONDS_THRESHOLD` | `3.0` | Seconds in unauthorized lane before violation |
| `DB_PATH` | `database/Violations.db` | SQLite file location |

---

### `core/state.py` — WebState

The communication bridge between the pipeline background thread and the FastAPI web server. Uses standard Python threading primitives:

| Attribute | Type | Purpose |
|---|---|---|
| `frame_buffer` | `bytes` | Latest JPEG frame for MJPEG stream |
| `frame_lock` | `threading.Lock` | Protects frame_buffer from race conditions |
| `stop_event` | `threading.Event` | Set by /api/stop to terminate the loop |
| `pause_event` | `threading.Event` | Set/cleared by /api/pause and /api/resume |
| `confirm_event` | `threading.Event` | Set by /api/confirm to end calibration |
| `next_frame_event` | `threading.Event` | Set by /api/next_frame during calibration |
| `state` | `str` | Current lifecycle state for the dashboard |

`threading.Event` is used instead of shared booleans because `.set()` / `.wait()` / `.is_set()` are atomic — no mutex needed for the signal itself.

---

### `core/lane_detector.py` — Lane Detection

`detect_lanes_with_model(frame)`:
- Loads the custom YOLO segmentation model and runs `predict()` at `conf=0.3`.
- Reads `results[0].masks.xy` — pixel-coordinate polygon arrays (not bounding boxes).
- Splits polygons by class ID into `normal_polys` and `unauthorized_polys`.
- Returns both polygon lists (used throughout Phase 3) and an annotated frame (used during calibration).

`draw_lane_overlays(frame, normal_polys, unauthorized_polys)`:
- Draws transparent colored fills (`alpha=0.15`) and solid boundary lines.
- Called every frame in Phase 3 to keep lane outlines visible on the live stream.

---

### `core/plate_worker.py` — ALPR Background Thread

`parse_turkish_plate(text)`:
- Strips spaces, converts to uppercase.
- Applies regex: `^\d{2}[A-Z]{1,3}\d{2,4}$`
- Returns formatted string (`"34 ABC 1234"`) or `None` if it does not match.

`PlateRecognitionWorker(threading.Thread)`:
- Initializes `FastALPR` lazily inside `run()` (heavy model load happens in the background thread, not blocking the main loop startup).
- Reads from `crop_queue` in a blocking loop.
- Receives `(None, None)` as a sentinel to exit cleanly.
- Updates `best_plates[track_id]` only if the new confidence is higher than the current best.

---

### `core/violation_manager.py` — State Machine

Tracks per-vehicle state using wall-clock time:

```
safe ──(enters unauthorized)──▶ warning ──(≥ 3 seconds)──▶ violation
 ▲                                  │
 └──────(exits unauthorized)────────┘  (timer resets)
```

Once a vehicle reaches `violation`, it stays there permanently (the `_violated` set is never cleared during a session).

`transfer_state(old_id, new_id)` — used by the re-identification logic to move a vehicle's timer and violation flag to its new ByteTrack ID.

---

### `core/database.py` — SQLite Operations

Four clean functions hiding all raw SQL:

| Function | What it does |
|---|---|
| `init_db()` | Opens/creates the DB, creates the `violations` table if missing |
| `insert_violation(cur, ts, source, vtime)` | Inserts a new row with `"Scanning..."` as plate placeholder, returns `lastrowid` |
| `update_plate(cur, db_id, text, confidence)` | Overwrites plate text and confidence for an existing row |
| `finalize_unresolved(cur, db_id)` | Changes `"Scanning..."` to `"Unreadable"` if OCR never succeeded |
| `get_violation_count(cur, source)` | Returns total violations for a given video file |

---

### `api/routes_pipeline.py` — Pipeline Control

| Endpoint | Method | What it does |
|---|---|---|
| `/api/start` | POST | Stops existing pipeline, creates WebState, launches new thread |
| `/api/stop` | POST | Sets `stop_event` — pipeline exits at next loop iteration |
| `/api/pause` | POST | Sets `pause_event` — loop waits in a sleep-loop |
| `/api/resume` | POST | Clears `pause_event` — loop continues |

Holds a module-level `_state` dict (`{"thread": ..., "web_state": ...}`) so all route handlers share the same pipeline reference without circular imports.

---

### `api/routes_calibration.py` — Calibration Control

| Endpoint | Method | What it does |
|---|---|---|
| `/api/confirm` | POST | Sets `confirm_event` → calibration loop exits, Phase 3 starts |
| `/api/next_frame` | POST | Sets `next_frame_event` → calibration loop skips to next frame |

---

### `api/routes_data.py` — Read-only Data

| Endpoint | Method | What it does |
|---|---|---|
| `/api/status` | GET | Returns current state, FPS, frame count |
| `/api/videos` | GET | Lists `.mp4` files in `input_videos/` |
| `/api/violations` | GET | Returns 50 most recent DB rows |
| `/api/stats` | GET | Returns total / readable / unreadable counts |
| `/video_feed` | GET | MJPEG multipart stream — consumed by `<img>` tag in dashboard |

The MJPEG stream generator reads `web_state.frame_buffer` in a loop with a 33ms sleep (~30 fps), yielding multipart JPEG chunks. The browser `<img>` tag renders these as a live video without any JavaScript video decoding.

---

## 5. Technologies Used and Why

| Technology | Role | Why this choice |
|---|---|---|
| **Ultralytics YOLOv8** | Lane segmentation + vehicle detection | Single library for both tasks; supports segmentation masks natively; ByteTrack is built in |
| **ByteTrack** | Multi-object vehicle tracking | Maintains persistent IDs across frames even under occlusion; built into Ultralytics |
| **FastALPR** | License plate detection + OCR | Lightweight ONNX-based pipeline; runs on CPU without GPU; modular detector/OCR |
| **ONNX Runtime** | FastALPR inference backend | Hardware-independent; faster than pure Python inference |
| **FastAPI** | Web server + REST API | Async-native; `StreamingResponse` handles MJPEG without extra config; auto-generates API docs |
| **Uvicorn** | ASGI server | Standard companion for FastAPI; single command to run |
| **OpenCV** | Video I/O, drawing, encoding | Industry standard for frame-level video processing; `pointPolygonTest`, `addWeighted`, `imencode` all in one library |
| **SQLite** | Violation storage | Zero configuration, no external server, file-based — ideal for single-machine deployment |
| **threading** | ALPR parallelism | ALPR inference must not block the main tracking loop; Python threading is sufficient since ALPR releases the GIL during ONNX inference |

---

## 6. Critical Design Decisions

**Why is ALPR in a separate thread?**
`yolo_model.track()` is a blocking call that takes most of each frame's budget. Running FastALPR in the same thread would cut FPS by 60–80%. The background thread reads from a queue at its own pace without blocking the main loop.

**Why 3 seconds for the violation threshold?**
This mirrors the behavior of real EDS (Electronic Detection System) enforcement cameras in Turkey. A shorter threshold would generate false positives from vehicles briefly drifting lanes. A longer one would miss short violations.

**Why store `best_plates` as a dict keyed by `track_id`?**
The same vehicle is cropped and sent to ALPR many times across different frames. The dict keeps only the single highest-confidence result, so the final plate written to the DB is always the clearest reading — not the most recent one.

**Why does `insert_violation()` write `"Scanning..."` first?**
ALPR runs asynchronously in the background thread. The violation is confirmed before the plate is read. Writing a placeholder row immediately ensures the violation timestamp is accurate, and `update_plate()` fills in the real plate text as soon as OCR completes.

**Why use the bottom-center point for `pointPolygonTest`?**
The bottom-center of the bounding box corresponds to the vehicle's road contact point when viewed from an overhead or rear-angle camera. This single point provides a clear, noise-free signal for whether the vehicle is in an unauthorized lane.

**Why `threading.Event` instead of shared booleans?**
Events are atomic: `.set()` and `.is_set()` have no race conditions. A plain `bool` would require a separate `threading.Lock` to be safe. Events also support `.wait()` with a timeout, which is used in the pause loop.

**Why rewind to frame 0 after calibration?**
Lane calibration reads frames from the video to find a clear shot of the lanes — often skipping to frame 30 or 60. After calibration, the video is reset to frame 0 so tracking covers the full footage including the early frames.

---

## 7. Database Schema

```sql
CREATE TABLE violations (
    vehicle_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME,   -- Wall-clock time when violation was first detected
    input_source    TEXT,       -- Video filename (e.g. "video1.mp4")
    violation_time  TEXT,       -- Position in the video (e.g. "2:17")
    license_plate   TEXT,       -- Plate text, or "Scanning..." / "Unreadable"
    ALPR_confidence REAL        -- Average per-character OCR confidence (0.0–1.0)
)
```

**Lifecycle of a row:**

```
insert_violation() → license_plate = "Scanning...", ALPR_confidence = 0.0
     ↓  (ALPR thread reads plate)
update_plate()     → license_plate = "34 ABC 1234", ALPR_confidence = 0.97
     ↓  (if OCR never succeeded)
finalize_unresolved() → license_plate = "Unreadable"
```

---

## 8. Configuration Parameters

All in `core/config.py`. Edit this file to tune the system without touching any other code.

| Parameter | Default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `PLATE_LOCKED_THRESHOLD` | `0.95` | More ALPR attempts per vehicle (higher accuracy) | Fewer attempts (faster, lower accuracy) |
| `VIOLATION_SECONDS_THRESHOLD` | `3.0` | Fewer violations logged (stricter) | More violations logged (more sensitive) |
| `VEHICLE_CLASSES` | `[2,5,7]` | — | Remove a class to ignore that vehicle type |

Pipeline-level tuning in `pipeline.py`:

| Parameter | Default | Effect |
|---|---|---|
| `PROFILE_MODE` | `True` | Set to `False` to disable FPS CSV logging entirely |
| `PROFILE_WARMUP_SECONDS` | `10` | Seconds of tracking skipped before measurement starts |
| `PROFILE_DURATION_SECONDS` | `60` | Seconds of measurement before auto-stop |

---

## 9. Installation and Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

If using the local FastALPR submodule:
```bash
pip install -e ./fast-alpr
```

### Run

```bash
python dashboard.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

### Workflow

1. Select a video from the dropdown.
2. Click **Start** — lane detection runs automatically on the first frame.
3. Review the detected lane polygons on screen.
   - Click **Next Frame** to re-run detection on a later frame if the current one is unclear.
   - Click **Confirm** when the lane boundaries look correct.
4. Tracking begins immediately. The live video stream shows:
   - Green box + green lane fill → vehicle in a normal lane
   - Orange box → vehicle in unauthorized lane, timer running
   - Red box + `VIOLATION!` label → violation recorded
   - Plate text + confidence shown at bottom of each box
5. Violations appear in the table on the right in real time.
6. Click **Stop** to end the session. Any plates still unread are marked `Unreadable` in the database.
