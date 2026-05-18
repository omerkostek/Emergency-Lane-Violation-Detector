# dashboard.py
# ─────────────────────────────────────────────────────────────
# Application entry point.
# Creates the FastAPI app, mounts all route groups, and serves
# the HTML template. All real logic lives in api/ and core/.
#
# Run:  python dashboard.py

import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.routes_pipeline import router as pipeline_router
from api.routes_calibration import router as calibration_router
from api.routes_data import router as data_router

# App setup 
app = FastAPI(title="Emergency Lane Violation Detector")

os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Mount route groups 
app.include_router(pipeline_router)
app.include_router(calibration_router)
app.include_router(data_router)


# HTML page 
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the single-page dashboard."""
    return templates.TemplateResponse(name="index.html", request=request)


# Startup 
if __name__ == "__main__":
    print("\n  Emergency Lane Violation Detector")
    print("  http://127.0.0.1:8001\n")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
