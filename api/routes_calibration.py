# api/routes_calibration.py
# ─────────────────────────────────────────────────────────────
# Calibration phase endpoints: confirm lane detection and
# request next frame for a better calibration shot.
# These routes are only meaningful while state == "calibrating".
# ─────────────────────────────────────────────────────────────

from fastapi import APIRouter
from api.routes_pipeline import get_web_state

router = APIRouter()


@router.post("/api/confirm")
async def confirm_calibration():
    """User has reviewed the detected lanes and wants to start tracking.
    
    Sets the confirm_event which unblocks the calibration wait-loop
    in pipeline.py and advances to Phase 3.
    """
    ws = get_web_state()
    if ws:
        ws.confirm_event.set()
    return {"status": "confirmed"}


@router.post("/api/next_frame")
async def next_frame_calibration():
    """Skip to the next frame (actually +10 frames) for lane re-detection.
    
    Sets next_frame_event which causes the calibration loop to break
    out of its streaming wait and read the next batch of frames.
    """
    ws = get_web_state()
    if ws and ws.state == "calibrating":
        ws.next_frame_event.set()
    return {"status": "next_frame"}
