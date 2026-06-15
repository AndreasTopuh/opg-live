"""
OPG-Live MVP — FastAPI backend.

Endpoints:
  GET  /              -> serves the frontend (frontend/index.html)
  POST /api/analyze   -> upload OPG image -> overview PNG + findings (bbox/FDI/disease)
  POST /api/explain   -> {id, idx, arm} -> GPT-4o L-F-V explanation (click-to-explain)
  GET  /api/health    -> status

Run (Colab): uvicorn main:app --host 0.0.0.0 --port 8000
DRIVE env var points to the Google Drive opg-live folder with checkpoints.
"""
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import OPGPipeline

DRIVE = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/opg-live")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DIST = os.path.join(_ROOT, "frontend", "dist")        # built Vite app (npm run build)
STATIC = os.path.join(_HERE, "static")                # vanilla fallback (no Node)

app = FastAPI(title="OPG-Live MVP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# serve built Vite assets if present (prod / Colab after `npm run build`)
if os.path.isdir(os.path.join(DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

_pipe = None  # lazy-loaded on first request (so the server starts instantly)


def pipe():
    global _pipe
    if _pipe is None:
        _pipe = OPGPipeline(DRIVE)
    return _pipe


@app.get("/")
def index():
    # prefer the built Vite SPA; fall back to the vanilla page (no Node needed).
    # In local dev you usually hit the Vite dev server (:5173) instead, which
    # proxies /api here — so this route mainly serves prod/Colab.
    for p in (os.path.join(DIST, "index.html"), os.path.join(STATIC, "index.html")):
        if os.path.exists(p):
            return FileResponse(p)
    return {"msg": "No frontend built. Run `npm run dev` (local) or `npm run build`."}


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None, "drive": DRIVE}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "could not decode image"}
    return pipe().analyze(img)


class SegmentReq(BaseModel):
    id: str
    idx: int


@app.post("/api/segment")
def segment(req: SegmentReq):
    return pipe().segment(req.id, req.idx)


class ExplainReq(BaseModel):
    id: str
    idx: int
    arm: str = "hybrid"


@app.post("/api/explain")
def explain(req: ExplainReq):
    return pipe().explain(req.id, req.idx, req.arm)
