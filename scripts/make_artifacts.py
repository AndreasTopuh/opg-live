"""
Generate 3-arm grounding artifacts untuk eksperimen faithfulness (Stage 2 -> Stage 3).

Untuk tiap lesion (deteksi GT DENTEX: bbox + kelas), pakai SAM+adapter hasilkan
lesion mask, lalu render 3 representasi grounding yang dikirim ke GPT-4o:
  - bbox   : OPG + kotak bbox
  - mask   : OPG + overlay mask lesi
  - hybrid : OPG + kotak + overlay mask

Output (ke Drive):
  outputs/artifacts/{bbox,mask,hybrid}/{lesion_id}.png
  outputs/artifacts/manifest.jsonl   (1 baris/lesion: id, kelas, bbox, path 3 arm, area)

Pakai EVAL SET stratified (n per kelas) supaya biaya GPT-4o terkendali.
HierarchicalDet (deteksi prediksi) bisa di-swap di sini untuk demo deployment;
untuk eksperimen terkontrol kita pakai deteksi GT (isolasi variabel grounding).
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

from dentex_dataset import (
    CLASS_NAMES,
    load_records,
    sample_per_class,
)
from sam_adapter import inject_adapters, load_adapter_state

BOX_COLOR = (0, 255, 0)      # hijau (BGR)
MASK_COLOR = (0, 0, 255)     # merah (BGR)
ALPHA = 0.45


def load_model(sam_ckpt, adapter_ckpt, device):
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    inject_adapters(sam)
    state = torch.load(adapter_ckpt, map_location="cpu")
    load_adapter_state(sam, state["state"])
    print(f"Adapter loaded (train Dice {state.get('dice', '?')}, epoch {state.get('epoch', '?')})")
    sam.to(device).eval()
    return SamPredictor(sam)


def draw_bbox(img, box):
    out = img.copy()
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(out, (x0, y0), (x1, y1), BOX_COLOR, 3)
    return out


def draw_mask(img, mask):
    out = img.copy()
    color = np.zeros_like(img)
    color[mask > 0] = MASK_COLOR
    return cv2.addWeighted(color, ALPHA, out, 1.0, 0)


def draw_hybrid(img, box, mask):
    return draw_bbox(draw_mask(img, mask), box)


def run(args):
    device = "cuda"
    js = glob.glob(f"{args.drive}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    recs = load_records(js, xr)
    evalset = sample_per_class(recs, args.n_per_class, seed=args.seed)
    print(f"Total lesion {len(recs)} | eval set {len(evalset)} ({args.n_per_class}/kelas)")

    out_dir = f"{args.drive}/outputs/artifacts"
    for arm in ["bbox", "mask", "hybrid"]:
        os.makedirs(f"{out_dir}/{arm}", exist_ok=True)

    predictor = load_model(args.sam_ckpt, args.adapter, device)

    # group per gambar -> set_image (encode ViT-H) sekali per gambar
    by_img = defaultdict(list)
    for r in evalset:
        by_img[r["img_path"]].append(r)

    manifest = []
    for img_path, lesions in by_img.items():
        img_bgr = cv2.imread(img_path)             # HxWx3 BGR
        predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        for r in lesions:
            x, y, bw, bh = r["bbox_xywh"]
            box = np.array([x, y, x + bw, y + bh])
            with torch.no_grad():
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(np.uint8)       # HxW {0,1}

            lid = f"{r['image_id']}_{r['ann_id']}"
            arts = {
                "bbox": draw_bbox(img_bgr, box),
                "mask": draw_mask(img_bgr, mask),
                "hybrid": draw_hybrid(img_bgr, box, mask),
            }
            for arm, im in arts.items():
                cv2.imwrite(f"{out_dir}/{arm}/{lid}.png", im)

            manifest.append({
                "lesion_id": lid,
                "img_file": r["img_file"],
                "cls": r["cls"],
                "cls_name": CLASS_NAMES[r["cls"]],
                "bbox_xyxy": [float(v) for v in box],
                "mask_area_px": int(mask.sum()),
                "sam_score": float(scores[0]),
                "artifacts": {arm: f"{arm}/{lid}.png" for arm in arts},
            })
        print(f"  {os.path.basename(img_path)}: {len(lesions)} lesion")

    with open(f"{out_dir}/manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"\n✅ {len(manifest)} lesion × 3 arm -> {out_dir}")
    print(f"   manifest: {out_dir}/manifest.jsonl")
    # ringkasan per kelas
    per = defaultdict(int)
    for m in manifest:
        per[m["cls_name"]] += 1
    print("   per kelas:", dict(per))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--sam_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--adapter", default="/content/drive/MyDrive/opg-live/checkpoints/adapter_best.pth")
    ap.add_argument("--n_per_class", type=int, default=40)  # 40×4 = 160 lesion eval
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
