"""
OPG-Live MVP pipeline: load the trained checkpoints and run an uploaded OPG
through Stage 1 (YOLO disease + YOLO FDI) -> Stage 2 (SAM+adapter mask) ->
Stage 3 (optional GPT-4o explanation per click).

Designed to run on Colab (GPU + Drive checkpoints). Analysis results are cached
in-memory by id so the click-to-explain endpoint reuses the computed masks.
"""
import base64
import os
import sys
import uuid

import cv2
import numpy as np
import torch

# make stage1/stage2/stage3 modules importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # OPG-Live/
for _d in ("stage1", "stage2", "stage3"):
    sys.path.insert(0, os.path.join(_ROOT, "scripts", _d))

from fdi_assign import EnumFDI                                # stage1
from make_artifacts import draw_bbox, draw_hybrid, draw_mask, load_sam  # stage3 helpers
from make_overview import COLORS                             # stage3 (per-class colours)

CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}

# in-memory cache of analyses: id -> {"img": bgr, "dets": [...]}
_CACHE = {}


def _png_b64(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


class OPGPipeline:
    def __init__(self, drive, conf=0.3, imgsz=1024, device="cuda"):
        from ultralytics import YOLO

        ckpt = f"{drive}/checkpoints"
        self.conf = conf
        self.imgsz = imgsz
        self.disease = YOLO(f"{ckpt}/yolov8_dentex.pt")
        self.enum = EnumFDI(self._first_existing(
            f"{ckpt}/yolov8_enum.pt", f"{ckpt}/yolo8_enum.pt"), imgsz, conf)
        self.predictor = load_sam(f"{ckpt}/sam_vit_h_4b8939.pth",
                                  f"{ckpt}/adapter_best.pth", device)
        self.drive = drive
        self._retr = None
        self._client = None
        print("OPGPipeline ready (YOLO disease + YOLO FDI + SAM adapter loaded).")

    @staticmethod
    def _first_existing(*paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return paths[0]

    # ---------- Stage 1+2: analyse an uploaded OPG ----------
    def analyze(self, img_bgr):
        res = self.disease.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        # tooth (FDI) detections — predict on the array directly
        eres = self.enum.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        teeth = [([float(v) for v in b.xyxy[0].tolist()], str(eres.names[int(b.cls)]), float(b.conf))
                 for b in eres.boxes]

        self.predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        overlay = img_bgr.copy()
        dets = []
        for i, b in enumerate(res.boxes):
            cls = int(b.cls)
            conf = float(b.conf)
            box = [int(v) for v in b.xyxy[0].tolist()]
            with torch.no_grad():
                masks, scores, _ = self.predictor.predict(box=np.array(box), multimask_output=False)
            mask = masks[0].astype(np.uint8)
            fdi = EnumFDI.assign(box, teeth)
            overlay[mask > 0] = COLORS.get(cls, (0, 255, 0))
            dets.append({"idx": i, "cls": cls, "disease": CLASS_NAMES[cls], "conf": round(conf, 3),
                         "bbox": box, "fdi": fdi, "mask": mask, "sam_score": round(float(scores[0]), 3)})

        # blended overview + labels
        ov = cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)
        for d in dets:
            x0, y0, x1, y1 = d["bbox"]
            color = COLORS.get(d["cls"], (0, 255, 0))
            cv2.rectangle(ov, (x0, y0), (x1, y1), color, 2)
            lbl = (f"Q:{d['fdi'][0]} N:{d['fdi'][1]} " if d["fdi"] and len(d["fdi"]) == 2 else "") + \
                  f"{d['disease']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(ov, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
            cv2.putText(ov, lbl, (x0 + 2, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        aid = uuid.uuid4().hex[:8]
        _CACHE[aid] = {"img": img_bgr, "dets": dets}
        findings = [{"idx": d["idx"], "disease": d["disease"], "fdi": d["fdi"],
                     "conf": d["conf"], "bbox": d["bbox"]} for d in dets]
        return {"id": aid, "overview": _png_b64(ov), "findings": findings}

    # ---------- Stage 3: explain one finding (click-to-explain) ----------
    def explain(self, aid, idx, arm="hybrid"):
        if aid not in _CACHE:
            return {"error": "analysis expired; re-upload"}
        entry = _CACHE[aid]
        d = next((x for x in entry["dets"] if x["idx"] == idx), None)
        if d is None:
            return {"error": "finding not found"}

        img, box, mask = entry["img"], np.array(d["bbox"]), d["mask"]
        art = {"bbox": draw_bbox(img, box), "mask": draw_mask(img, mask),
               "hybrid": draw_hybrid(img, box, mask)}[arm]

        # lazy RAG + GPT-4o (optional; needs OPENROUTER_API_KEY + KB embeddings)
        if not os.environ.get("OPENROUTER_API_KEY"):
            return {"artifact": _png_b64(art), "error": "set OPENROUTER_API_KEY for explanation"}
        try:
            if self._retr is None:
                from retriever import Retriever
                from llm_gpt import get_client
                self._retr = Retriever(f"{self.drive}/data/kb")
                self._client = get_client()
            from prompt_builder import build_prompt
            from llm_gpt import explain as gpt_explain
            import tempfile

            chunks = self._retr.search(
                f"{d['disease']} on panoramic dental radiograph: appearance and management", k=4)
            tmp = os.path.join(tempfile.gettempdir(), f"{aid}_{idx}_{arm}.png")
            cv2.imwrite(tmp, art)
            prompt = build_prompt(arm, d["disease"], d["fdi"], chunks, f"{aid}_{idx}")
            parsed, raw = gpt_explain(self._client, prompt, tmp)
            return {"artifact": _png_b64(art), "arm": arm,
                    "findings_json": parsed, "raw": raw,
                    "retrieved": [c["id"] for c in chunks]}
        except Exception as e:
            return {"artifact": _png_b64(art), "error": f"explain failed: {e}"}
