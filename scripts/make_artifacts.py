"""
Generate 3-arm grounding artifacts untuk eksperimen faithfulness (Stage 1 -> 2 -> 3).

Metodologi: "same diagnosis from Stage 1 feeds all arms" (draft §4.3). Untuk tiap
DETEKSI Stage 1 (YOLOv8: bbox + diagnosis), bbox YANG SAMA dipakai untuk:
  - bbox   : OPG + kotak
  - mask   : OPG + overlay mask (SAM+adapter, box prompt = bbox YOLO)
  - hybrid : OPG + kotak + mask
Hanya spatial referent yang berubah -> perbandingan terkontrol.

Mode:
  --mode yolo  (default) : deteksi prediksi YOLO Stage 1 (sesuai metodologi)
  --mode gt              : deteksi GT DENTEX (untuk analisis sekunder/oracle)

Output (Drive):
  outputs/artifacts/{bbox,mask,hybrid}/{det_id}.png
  outputs/artifacts/manifest.jsonl
    (det_id, img_file, pred_cls, pred_cls_name, conf, bbox, mask_area, sam_score,
     matched_gt_cls, gt_iou, correct)   # field GT hanya untuk evaluasi, bukan input
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

from dentex_dataset import CLASS_NAMES, load_records, sample_per_class
from sam_adapter import inject_adapters, load_adapter_state

BOX_COLOR = (0, 255, 0)      # hijau (BGR)
MASK_COLOR = (0, 0, 255)     # merah (BGR)
ALPHA = 0.45


# ---------- model ----------
def load_sam(sam_ckpt, adapter_ckpt, device):
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    inject_adapters(sam)
    state = torch.load(adapter_ckpt, map_location="cpu")
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
    """file_name -> list of (bbox_xyxy, cls). Untuk matching evaluasi."""
    d = json.load(open(disease_json))
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    by_file = defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        by_file[id2file[a["image_id"]]].append(([x, y, x + w, y + h], a["category_id_3"]))
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
    print(f"YOLO: {len(dets)} deteksi dari {len(imgs)} gambar (conf>{conf})")
    return dets


def gt_detections(disease_json, xrays_dir, n_per_class, seed):
    """Mode oracle: pakai lesi GT sebagai 'deteksi'."""
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
    print(f"Eval set: {len(dets)} deteksi ({args.n_per_class}/kelas, mode={args.mode})")

    gt_idx = build_gt_index(js)   # untuk matching evaluasi

    out_dir = f"{args.drive}/outputs/artifacts"
    for arm in ["bbox", "mask", "hybrid"]:
        os.makedirs(f"{out_dir}/{arm}", exist_ok=True)

    predictor = load_sam(args.sam_ckpt, args.adapter, device)

    by_img = defaultdict(list)
    for dct in dets:
        by_img[dct["img_path"]].append(dct)

    manifest = []
    for img_path, group in by_img.items():
        img_bgr = cv2.imread(img_path)
        predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
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

            # match ke GT (IoU>=0.5) -> tahu diagnosis Stage 1 benar/tidak
            gt_cls, gt_iou = None, 0.0
            for gb, gc in gt_idx.get(dct["img_file"], []):
                i = iou_xyxy(dct["bbox_xyxy"], gb)
                if i > gt_iou:
                    gt_iou, gt_cls = i, gc
            matched = gt_cls if gt_iou >= 0.5 else None

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
    print(f"\n✅ {len(manifest)} deteksi × 3 arm -> {out_dir}")
    print(f"   per kelas (pred): {dict(per)}")
    if args.mode == "yolo":
        print(f"   diagnosis benar (match GT): {n_correct}/{len(manifest)} "
              f"({100*n_correct/max(1,len(manifest)):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--mode", choices=["yolo", "gt"], default="yolo")
    ap.add_argument("--yolo_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/yolov8_dentex.pt")
    ap.add_argument("--images_dir", default="/content/yolo/images/val")  # held-out
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--sam_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--adapter", default="/content/drive/MyDrive/opg-live/checkpoints/adapter_best.pth")
    ap.add_argument("--n_per_class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
