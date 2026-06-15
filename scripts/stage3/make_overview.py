"""
Overview renderer (Phase C / demo): ALL YOLO detections on one OPG at once,
colour-coded per disease class.

Two modes:
  default          : bbox + label
  --masks          : SAM+adapter MASK overlay per lesion (colour per disease) + box

Difference from make_artifacts.py:
  - make_artifacts = ONE lesion per image (for the faithfulness experiment)
  - make_overview  = ALL lesions per image (clinician summary view)

Output: outputs/overview/{file}.png
"""
import argparse
import glob
import os

import cv2
import numpy as np

# make stage1/stage2/stage3 modules importable regardless of cwd
import sys
_S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/
for _d in ("stage1", "stage2", "stage3"):
    sys.path.insert(0, os.path.join(_S, _d))

from dentex_dataset import CLASS_NAMES  # stage2

# Colour per class (BGR for OpenCV)
COLORS = {
    0: (255, 0, 0),     # Impacted    - blue
    1: (255, 255, 0),   # Caries      - cyan
    2: (0, 255, 255),   # Periapical  - yellow
    3: (0, 255, 0),     # Deep Caries - green
}
ALPHA = 0.45


def run(args):
    from ultralytics import YOLO

    model = YOLO(args.yolo_ckpt)

    predictor = None
    if args.masks:
        from make_artifacts import load_sam
        predictor = load_sam(args.sam_ckpt, args.adapter, "cuda")

    enum = None
    if args.enum_ckpt:
        from fdi_assign import EnumFDI
        enum = EnumFDI(args.enum_ckpt, args.imgsz, args.conf)

    out_dir = f"{args.drive}/outputs/overview"
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(f"{args.images_dir}/*.png")) + sorted(glob.glob(f"{args.images_dir}/*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]

    for p in imgs:
        img = cv2.imread(p)
        res = model.predict(p, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = [(int(b.cls), float(b.conf), [int(v) for v in b.xyxy[0].tolist()]) for b in res.boxes]

        # mask overlay (if --masks)
        if predictor is not None and dets:
            import torch
            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            overlay = img.copy()
            for cls, conf, box in dets:
                with torch.no_grad():
                    masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
                overlay[masks[0] > 0] = COLORS.get(cls, (0, 255, 0))
            img = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

        # FDI per detection (if enum active)
        teeth = enum.teeth(p) if enum else []

        # boxes + labels on top
        for cls, conf, box in dets:
            x0, y0, x1, y1 = box
            color = COLORS.get(cls, (0, 255, 0))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            fdi = enum.assign(box, teeth) if enum else None
            # GT-style DENTEX label: "Q: <quadrant> N: <tooth> D: <disease>"
            if fdi and len(str(fdi)) == 2:
                q, n = str(fdi)[0], str(fdi)[1]
                label = f"Q: {q} N: {n} D: {CLASS_NAMES[cls]}"
            elif fdi:
                label = f"FDI {fdi} {CLASS_NAMES[cls]}"
            else:
                label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
            cv2.putText(img, label, (x0 + 2, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imwrite(f"{out_dir}/{os.path.basename(p)}", img)

    print(f"OK: {len(imgs)} overview ({'mask' if args.masks else 'bbox'}) -> {out_dir}")
    print("   Colours: Impacted=blue, Caries=cyan, Periapical=yellow, Deep Caries=green")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--yolo_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/yolov8_dentex.pt")
    ap.add_argument("--images_dir", default="/content/yolo/images/val")
    ap.add_argument("--masks", action="store_true", help="overlay SAM mask per lesion")
    ap.add_argument("--enum_ckpt", default="", help="tooth detector -> show FDI in label")
    ap.add_argument("--sam_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--adapter", default="/content/drive/MyDrive/opg-live/checkpoints/adapter_best.pth")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
