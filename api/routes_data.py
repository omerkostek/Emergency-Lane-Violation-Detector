# 
# Read-only data endpoints: pipeline status, video list,
# statistics, and recent violations from the database.
# These routes never mutate the pipeline or the database.
# 

import os
import sqlite3
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.config import INPUT_DIR, DB_PATH
from api.routes_pipeline import get_web_state

router = APIRouter()

# Database helper (read-only copy, dashboard-side)
def _get_db():
    """Open a read-only connection to the violations database.
    Returns None if the database file doesn't exist yet."""
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # Row objects behave like dicts
    return conn

# Pipeline status
@router.get("/api/status")
async def get_status():
    """Return the current pipeline lifecycle state and runtime metrics.
    
    States: idle → detecting → calibrating → tracking → paused → done
    """
    ws = get_web_state()
    if ws is None:
        return {"state": "idle", "frame_width": 0, "frame_height": 0,
                "fps": 0, "frame_count": 0}

    # Report "paused" when tracking is frozen (pause_event is set)
    current_state = ws.state
    if current_state == "tracking" and ws.pause_event.is_set():
        current_state = "paused"

    return {
        "state": current_state,
        "frame_width": ws.frame_width,
        "frame_height": ws.frame_height,
        "fps": round(ws.fps, 1),
        "frame_count": ws.frame_count,
    }

# Video list
@router.get("/api/videos")
async def list_videos():
    """Return sorted list of .mp4 filenames available in the input directory."""
    if not os.path.isdir(INPUT_DIR):
        return []
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.mp4'))
    return files

# Statistics
@router.get("/api/stats")
async def get_statistics():
    """Aggregate counts: total violations, readable plates, unreadable plates."""
    conn = _get_db()
    if conn is None:
        return {"total_violations": 0, "readable_plates": 0, "ghost_vehicles": 0}

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM violations")
    total = cur.fetchone()[0]

    # "Readable" means the OCR returned a valid plate (not a placeholder)
    cur.execute(
        "SELECT COUNT(*) FROM violations "
        "WHERE license_plate != 'Unreadable' AND license_plate != 'Scanning...'"
    )
    readable = cur.fetchone()[0]
    conn.close()

    return {
        "total_violations": total,
        "readable_plates": readable,
        "ghost_vehicles": total - readable,
    }

# Violations list
@router.get("/api/violations")
async def get_recent_violations():
    """Return the 50 most recent violation records, newest first."""
    conn = _get_db()
    if conn is None:
        return []

    cur = conn.cursor()
    cur.execute("SELECT * FROM violations ORDER BY vehicle_id DESC LIMIT 50")
    rows = cur.fetchall()
    conn.close()

    return [dict(row) for row in rows]

# MJPEG video stream
def _generate_mjpeg():
    """Generator that yields MJPEG (Motion JPEG) multipart frames from the pipeline buffer.
    
    Stops automatically when the pipeline reaches the 'done' state.
    """
    while True:
        ws = get_web_state()

        if ws is not None and ws.frame_buffer is not None:
            with ws.frame_lock:
                frame = ws.frame_buffer
            if frame:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )

        # Send one final frame when the pipeline finishes, then close the stream
        if ws is not None and ws.state == "done":
            if ws.frame_buffer:
                with ws.frame_lock:
                    frame = ws.frame_buffer
                if frame:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                    )
            break

        time.sleep(0.033)

@router.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint consumed by the <img> tag in index.html."""
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"  #format for MJPEG streams
    )
