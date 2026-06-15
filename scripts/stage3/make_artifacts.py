"""
Generate 3-arm grounding artifacts for the faithfulness experiment (Stage 1 -> 2 -> 3).

Methodology: "same diagnosis from Stage 1 feeds all arms" (draft §4.3). For each
Stage-1 DETECTION (YOLOv8: bbox + diagnosis), the SAME bbox is used for:
  - bbox   : OPG + box
  - mask   : OPG + mask overlay (SAM+adapter, box prompt = YOLO bbox)
  - hybrid : OPG + box + mask
Only the spatial referent changes -> controlled comparison.

Modes:
  --mode yolo  (default) : predicted YOLO Stage-1 detections (per methodology)
  --mode gt              : GT DENTEX detections (secondary/oracle analysis)

Output (Drive):
  outputs/artifacts/{bbox,mask,hybrid}/{det_id}.png
  outputs/artifacts/manifest.jsonl
    (det_id, img_file, pred_cls, pred_cls_name, conf, bbox, mask_area, sam_score,
     matched_gt_cls, target_fdi, pred_fdi, gt_iou, correct)  # GT fields are for evaluation, not input
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

# make stage1/stage2/stage3 modules importable regardless of cwd
import sys
_S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/
for _d in ("stage1", "stage2", "stage3"):
    sys.path.insert(0, os.path.join(_S, _d))

from dentex_dataset import CLASS_NAMES, load_records, sample_per_class  # stage2
from fdi_assign import EnumFDI                                          # stage1
from sam_adapter import inject_adapters, load_adapter_state             # stage2

BOX_COLOR = (0, 255, 0)      # green (BGR)
MASK_COLOR = (0, 0, 255)     # red (BGR)
ALPHA = 0.45


# ---------- model ----------
def load_sam(sam_ckpt, adapter_ckpt, device):
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    inject_adapters(sam)
    state = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
    load_adapter_state(sam, state["state"])
    print(f"Adapter loaded (Dice {state.get('dice', '?')}, epoch {state.get('epoch', '?')})")
    sam.to(device).eval()
    return SamPredictor(sam)


# ---------- detections ----------
def iou_xyxy(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def build_gt_index(disease_json):
    """file_name -> list of (bbox_xyxy, cls3, fdi). For eval matching + target FDI."""
    d = json.load(open(disease_json))
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    cat1 = {c["id"]: str(c["name"]) for c in d.get("categories_1", [])}  # quadrant
    cat2 = {c["id"]: str(c["name"]) for c in d.get("categories_2", [])}  # tooth number
    by_file = defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        q = cat1.get(a.get("category_id_1"), "")
        t = cat2.get(a.get("category_id_2"), "")
        fdi = f"{q}{t}" if q and t else None       # two-digit FDI, e.g. "36"
        by_file[id2file[a["image_id"]]].append(([x, y, x + w, y + h], a["category_id_3"], fdi))
    return by_file


def yolo_detections(yolo_ckpt, images_dir, conf, imgsz):
    from ultralytics import YOLO

    model = YOLO(yolo_ckpt)
    dets = []
    imgs = sorted(glob.glob(f"{images_dir}/*.png")) + sorted(glob.glob(f"{images_dir}/*.jpg"))
    for img_path in imgs:
        res = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for i, b in enumerate(res.boxes):
            dets.append({
                "det_id": f"{stem}_{i}",
                "img_path": img_path,
                "img_file": os.path.basename(img_path),
                "bbox_xyxy": [float(v) for v in b.xyxy[0].tolist()],
                "pred_cls": int(b.cls),
                "conf": float(b.conf),
            })
    print(f"YOLO: {len(dets)} detections from {len(imgs)} images (conf>{conf})")
    return dets


def gt_detections(disease_json, xrays_dir, n_per_class, seed):
    """Oracle mode: use GT lesions as 'detections'."""
    recs = sample_per_class(load_records(disease_json, xrays_dir), n_per_class, seed)
    dets = []
    for r in recs:
        x, y, w, h = r["bbox_xywh"]
        dets.append({
            "det_id": f"{r['image_id']}_{r['ann_id']}",
            "img_path": r["img_path"],
            "img_file": r["img_file"],
            "bbox_xyxy": [x, y, x + w, y + h],
            "pred_cls": r["cls"],
            "conf": 1.0,
        })
    return dets


def sample_dets_per_class(dets, n_per_class, seed):
    by = defaultdict(list)
    for d in dets:
        by[d["pred_cls"]].append(d)
    rng = np.random.default_rng(seed)
    out = []
    for c, items in by.items():
        idx = rng.permutation(len(items))[:n_per_class]
        out += [items[i] for i in idx]
    return out


# ---------- render ----------
def draw_bbox(img, box):
    out = img.copy()
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(out, (x0, y0), (x1, y1), BOX_COLOR, 3)
    return out


def draw_mask(img, mask):
    color = np.zeros_like(img)
    color[mask > 0] = MASK_COLOR
    return cv2.addWeighted(color, ALPHA, img, 1.0, 0)


def draw_hybrid(img, box, mask):
    return draw_bbox(draw_mask(img, mask), box)


# ---------- main ----------
def run(args):
    device = "cuda"
    js = glob.glob(f"{args.drive}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")

    if args.mode == "yolo":
        dets = yolo_detections(args.yolo_ckpt, args.images_dir, args.conf, args.imgsz)
        dets = sample_dets_per_class(dets, args.n_per_class, args.seed)
    else:
        dets = gt_detections(js, xr, args.n_per_class, args.seed)
    print(f"Eval set: {len(dets)} detections ({args.n_per_class}/class, mode={args.mode})")

    gt_idx = build_gt_index(js)   # for eval matching

    out_dir = f"{args.drive}/outputs/artifacts"
    for arm in ["bbox", "mask", "hybrid"]:
        os.makedirs(f"{out_dir}/{arm}", exist_ok=True)

    predictor = load_sam(args.sam_ckpt, args.adapter, device)
    enum = EnumFDI(args.enum_ckpt, args.imgsz, args.conf) if args.enum_ckpt else None
    if enum:
        print("Enumeration YOLO active -> FDI predicted (pred_fdi)")

    by_img = defaultdict(list)
    for dct in dets:
        by_img[dct["img_path"]].append(dct)

    manifest = []
    for img_path, group in by_img.items():
        img_bgr = cv2.imread(img_path)
        predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        teeth = enum.teeth(img_path) if enum else []
        for dct in group:
            box = np.array(dct["bbox_xyxy"])
            with torch.no_grad():
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(np.uint8)

            arts = {
                "bbox": draw_bbox(img_bgr, box),
                "mask": draw_mask(img_bgr, mask),
                "hybrid": draw_hybrid(img_bgr, box, mask),
            }
            for arm, im in arts.items():
                cv2.imwrite(f"{out_dir}/{arm}/{dct['det_id']}.png", im)

            # match to GT (IoU>=0.5) -> is Stage-1 diagnosis correct? + target FDI
            gt_cls, gt_fdi, gt_iou = None, None, 0.0
            for gb, gc, gf in gt_idx.get(dct["img_file"], []):
                i = iou_xyxy(dct["bbox_xyxy"], gb)
                if i > gt_iou:
                    gt_iou, gt_cls, gt_fdi = i, gc, gf
            matched = gt_cls if gt_iou >= 0.5 else None
            target_fdi = gt_fdi if gt_iou >= 0.5 else None   # clean FDI for the experiment prompt
            pred_fdi = EnumFDI.assign(dct["bbox_xyxy"], teeth) if enum else None  # predicted FDI (deployment)

            manifest.append({
                "det_id": dct["det_id"],
                "img_file": dct["img_file"],
                "pred_cls": dct["pred_cls"],
                "pred_cls_name": CLASS_NAMES[dct["pred_cls"]],
                "conf": round(dct["conf"], 4),
                "bbox_xyxy": [round(v, 1) for v in dct["bbox_xyxy"]],
                "mask_area_px": int(mask.sum()),
                "sam_score": round(float(scores[0]), 4),
                "matched_gt_cls": matched,
                "target_fdi": target_fdi,
                "pred_fdi": pred_fdi,
                "fdi_correct": (pred_fdi == target_fdi) if (pred_fdi and target_fdi) else None,
                "gt_iou": round(gt_iou, 3),
                "correct": (matched == dct["pred_cls"]) if matched is not None else False,
                "artifacts": {arm: f"{arm}/{dct['det_id']}.png" for arm in arts},
            })

    with open(f"{out_dir}/manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    per = defaultdict(int)
    for m in manifest:
        per[m["pred_cls_name"]] += 1
    n_correct = sum(m["correct"] for m in manifest)
    print(f"\nOK: {len(manifest)} detections x 3 arms -> {out_dir}")
    print(f"   per class (pred): {dict(per)}")
    if args.mode == "yolo":
        print(f"   correct diagnosis (match GT): {n_correct}/{len(manifest)} "
              f"({100*n_correct/max(1,len(manifest)):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--mode", choices=["yolo", "gt"], default="yolo")
    ap.add_argument("--yolo_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/yolov8_dentex.pt")
    ap.add_argument("--enum_ckpt", default="", help="tooth detector (FDI). Empty = FDI from GT-match only")
    ap.add_argument("--images_dir", default="/content/yolo/images/val")  # held-out
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--sam_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--adapter", default="/content/drive/MyDrive/opg-live/checkpoints/adapter_best.pth")
    ap.add_argument("--n_per_class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
