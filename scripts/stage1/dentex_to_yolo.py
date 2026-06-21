"""
Convert DENTEX disease (hierarchical COCO) -> YOLOv8 format for Stage 1 detection.

Stage 1 = YOLO detector (4 disease classes), consistent with the supervisor's
panoramic-radiograph approach (Veerabhadrappa & Vengusamy, 2025, YOLOv7),
updated to YOLOv8. Official Plan B in the roadmap (HierarchicalDet/Detectron2
weights were never publicly released).

This is DATA PREP only (no architecture changes — Stage 1 is a baseline, not the
contribution). Beyond a plain COCO->YOLO format dump, for a rigorous baseline it:
  * class-stratified per-image split so the rare Periapical (n=158) keeps its
    proportion in val/test (a plain random split could starve it);
  * keeps OPGs with no disease as background images (empty-label negatives) to
    reduce false positives;
  * clamps bbox coords to [0,1] and skips degenerate boxes;
  * optionally writes a held-out test split (--test_frac) that is SEPARATE from
    the val set used for early-stopping (cleaner baseline reporting).

Output structure (default under /content for speed; weights later -> Drive):
  yolo/
    images/{train,val[,test]}/*.png
    labels/{train,val[,test]}/*.txt    # each line: "cls cx cy w h" (normalised 0-1)
    dentex.yaml

Classes (category_id_3 kept): 0 Impacted, 1 Caries, 2 Periapical, 3 Deep Caries
Split per-IMAGE (not per-lesion) to avoid lesion leakage across splits.
"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np

CLASS_NAMES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
}


def convert(args):
    js = glob.glob(f"{args.drive}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    d = json.load(open(js))

    images = {im["id"]: im for im in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # global class frequency -> each image's RAREST class is its stratification
    # key (a robust heuristic for multi-label / multi-disease images).
    cls_freq = Counter(a["category_id_3"] for a in d["annotations"])

    ann_ids = sorted(anns_by_img.keys())                       # images WITH disease
    bg_ids = [i for i in images if i not in anns_by_img]       # healthy-only -> background

    def strat_key(iid):
        return min((a["category_id_3"] for a in anns_by_img[iid]), key=lambda c: cls_freq[c])

    rng = np.random.default_rng(args.seed)
    train_ids, val_ids, test_ids = set(), set(), set()

    def split_group(ids):
        """Split one stratum independently so each class keeps its proportion."""
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * args.val_frac))
        n_test = int(round(n * args.test_frac))
        val_ids.update(ids[:n_val])
        test_ids.update(ids[n_val:n_val + n_test])
        train_ids.update(ids[n_val + n_test:])

    groups = defaultdict(list)
    for iid in ann_ids:
        groups[strat_key(iid)].append(iid)
    for ids in groups.values():
        split_group(ids)
    split_group(bg_ids)   # spread background images proportionally too

    splits = {"train": train_ids, "val": val_ids}
    if args.test_frac > 0:
        splits["test"] = test_ids

    for sp in splits:
        os.makedirs(f"{args.out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{sp}", exist_ok=True)

    n_box = 0
    per_split = {sp: Counter() for sp in splits}
    for sp, ids in splits.items():
        for iid in ids:
            im = images[iid]
            W, H = im["width"], im["height"]
            fn = im["file_name"]
            src = os.path.join(xr, fn)
            if not os.path.exists(src):
                continue
            shutil.copy(src, f"{args.out}/images/{sp}/{fn}")
            stem = os.path.splitext(fn)[0]
            lines = []
            for a in anns_by_img[iid]:                 # empty list for background images
                x, y, bw, bh = a["bbox"]
                cx, cy, w, h = (x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H
                if w <= 0 or h <= 0:
                    continue                           # skip degenerate boxes
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                w, h = min(w, 1.0), min(h, 1.0)
                lines.append(f"{a['category_id_3']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                per_split[sp][a["category_id_3"]] += 1
                n_box += 1
            with open(f"{args.out}/labels/{sp}/{stem}.txt", "w") as f:
                f.write("\n".join(lines))            # empty file = background negative

    yaml_path = f"{args.out}/dentex.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        if args.test_frac > 0:
            f.write("test: images/test\n")
        f.write("names:\n")
        for k in sorted(CLASS_NAMES):
            f.write(f"  {k}: {CLASS_NAMES[k]}\n")

    # transparency report (per-split per-class box counts -> use in baseline chapter)
    n_img = sum(len(ids) for ids in splits.values())
    print(f"Images: {n_img} (disease {len(ann_ids)} + background {len(bg_ids)}) | "
          + " | ".join(f"{sp} {len(ids)}" for sp, ids in splits.items()))
    for sp in splits:
        dist = {CLASS_NAMES[c]: per_split[sp][c] for c in sorted(CLASS_NAMES)}
        print(f"  {sp:5s} boxes: {dist}")
    print(f"OK: {n_box} boxes written. Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--out", default="/content/yolo")  # local Colab = fast
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.0)  # >0 -> add held-out test split
    ap.add_argument("--seed", type=int, default=42)
    convert(ap.parse_args())
