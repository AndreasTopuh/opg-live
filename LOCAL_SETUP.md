# Run the MVP locally (FastAPI + Vite/React)

Two processes: **backend** (FastAPI/Python, the models) and **frontend**
(Vite + React dev server). Frontend proxies `/api` → backend.

## ⚠️ Hardware reality (laptop GTX 1650 4 GB)
- **SAM ViT-H needs ~7-8 GB VRAM → won't fit on 4 GB.** Locally it runs on **CPU**
  (auto-detected). CPU encode is slow: **~30-90 s per uploaded image** (after that,
  clicking teeth is instant — masks are cached). YOLO stays fast.
- You need the checkpoints **on disk** (~2.6 GB total). Free up SSD if tight.
- For a snappy demo, prefer Colab (`notebooks/06_mvp_demo.ipynb`, GPU). Local is for
  development / offline showing.

## 0. Get the checkpoints locally
Download from Google Drive `opg-live/checkpoints/` into a local folder, e.g.
`C:/opg-data/checkpoints/`:
```
sam_vit_h_4b8939.pth     (2.5 GB)
adapter_best.pth         (~50 MB)
yolov8_dentex.pt         (~50 MB)
yolov8_enum.pt           (~50 MB)   # or yolo8_enum.pt
```
Also the disease JSON (for FDI fallback is optional). Set the folder as `DRIVE_ROOT`
(it expects `<DRIVE_ROOT>/checkpoints/...`). For explanations also copy `data/kb/`.

## 1. Backend (FastAPI)
```bash
cd OPG-Live/backend
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install fastapi uvicorn python-multipart ultralytics segment-anything pycocotools opencv-python torch torchvision numpy

# point to your local checkpoints folder
set DRIVE_ROOT=C:/opg-data          # Windows (cmd)   → expects C:/opg-data/checkpoints/...
# $env:DRIVE_ROOT="C:/opg-data"     # PowerShell
# force CPU explicitly if needed:  set OPG_DEVICE=cpu

uvicorn main:app --reload --port 8000
```
First request loads the models (slow on CPU). Health check: <http://localhost:8000/api/health>.

## 2. Frontend (Vite + React)
```bash
cd OPG-Live/frontend
npm install
npm run dev
```
Open <http://localhost:5173>. The dev server proxies `/api/*` to the backend on :8000.

## 3. Use it
Upload an OPG → **Overview** (all lesions, colour per disease + FDI) → click a tooth →
its **segmentation** (mask highlight + zoom + metrics). The "▦ Overview (all)" button
returns to the full view.

## Production / Colab (single server)
Build the SPA and let FastAPI serve it (no Node at runtime):
```bash
cd frontend && npm run build       # outputs frontend/dist/
# then just run uvicorn — main.py serves frontend/dist at "/"
```
On Colab without Node, `main.py` falls back to `backend/static/index.html` (the
vanilla single-file UI) — that's what `notebooks/06_mvp_demo.ipynb` uses.

## Notes
- `OPG_DEVICE=cpu|cuda` overrides auto-detection.
- The GPT-4o click-to-explain endpoint (`/api/explain`) exists but is off the current
  UI; the UI focuses on detect + segment (Stage 1+2).
