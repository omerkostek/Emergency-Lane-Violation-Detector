# api/routes_pipeline.py
# ─────────────────────────────────────────────────────────────
# Pipeline lifecycle endpoints: start, stop, pause, resume.
# These routes control the background video processing thread
# via the shared WebState object.
# ─────────────────────────────────────────────────────────────

import os
import threading

from fastapi import APIRouter
from pydantic import BaseModel

from core.config import INPUT_DIR
from core.state import WebState

router = APIRouter()


# ── Request model ─────────────────────────────────────────────
class StartRequest(BaseModel):
    video: str   # Filename relative to INPUT_DIR (e.g. "video1.mp4")


# ── Shared mutable pipeline state (set by start, read by all) ─
# We use a simple dict so the router module stays importable
# without circular imports (pipeline is imported lazily inside start).
_state: dict = {
    "thread": None,
    "web_state": None,
}


def get_web_state() -> WebState | None:
    """Return the current WebState; used by other route modules."""
    return _state["web_state"]


@router.post("/api/start")
async def start_pipeline(req: StartRequest):
    """Start or restart the video processing pipeline.
    
    Stops any currently running pipeline first, creates fresh WebState,
    then launches the pipeline in a daemon thread.
    """
    ws = _state["web_state"]

    # Gracefully stop an existing pipeline before starting a new one
    if ws is not None and not ws.stop_event.is_set():
        ws.stop_event.set()
        t = _state["thread"]
        if t and t.is_alive():
            t.join(timeout=5)

    # Import lazily to avoid circular dependency at module load time
    from pipeline import process_video_stream

    new_ws = WebState()
    _state["web_state"] = new_ws

    video_path = os.path.join(INPUT_DIR, req.video)

    def run():
        try:
            process_video_stream(video_path, True, new_ws)
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback; traceback.print_exc()
        finally:
            new_ws.state = "done"

    t = threading.Thread(target=run, daemon=True)
    _state["thread"] = t
    t.start()

    return {"status": "started"}


@router.post("/api/stop")
async def stop_pipeline():
    """Signal the pipeline to stop at the next opportunity."""
    ws = _state["web_state"]
    if ws:
        ws.stop_event.set()
    return {"status": "stopped"}


@router.post("/api/pause")
async def pause_pipeline():
    """Freeze the video stream (tracking loop waits in a sleep-loop)."""
    ws = _state["web_state"]
    if ws and ws.state == "tracking":
        ws.pause_event.set()
    return {"status": "paused"}


@router.post("/api/resume")
async def resume_pipeline():
    """Resume a paused video stream."""
    ws = _state["web_state"]
    if ws:
        ws.pause_event.clear()
    return {"status": "resumed"}
