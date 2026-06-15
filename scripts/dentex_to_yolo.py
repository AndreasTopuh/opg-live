"""
Convert DENTEX disease (hierarchical COCO) -> YOLOv8 format for Stage 1 detection.

Stage 1 = YOLO detector (4 disease classes), consistent with the supervisor's
panoramic-radiograph approach (Veerabhadrappa & Vengusamy, 2025, YOLOv7),
updated to YOLOv8. Official Plan B in the roadmap (HierarchicalDet/Detectron2
weights were never publicly released).

Output structure (default under /content for speed; weights later -> Drive):
  yolo/
    images/{train,val}/*.png
    labels/{train,val}/*.txt    # each line: "cls cx cy w h" (normalised 0-1)
    dentex.yaml

Classes (category_id_3 kept): 0 Impacted, 1 Caries, 2 Periapical, 3 Deep Caries
Split per-IMAGE (not per-lesion) to avoid lesion leakage across splits.
"""
import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np

CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}


def convert(args):
    js = glob.glob(f"{args.drive}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    d = json.load(open(js))

    images = {im["id"]: im for im in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # split per-image (deterministic)
    img_ids = sorted(anns_by_img.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(img_ids)
    n_val = int(len(img_ids) * args.val_frac)
    splits = {"val": set(img_ids[:n_val]), "train": set(img_ids[n_val:])}
    print(f"Images: {len(img_ids)} | train {len(splits['train'])} | val {len(splits['val'])}")

    for sp in ["train", "val"]:
        os.makedirs(f"{args.out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{sp}", exist_ok=True)

    n_box = 0
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
            for a in anns_by_img[iid]:
                x, y, bw, bh = a["bbox"]
                cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
                lines.append(f"{a['category_id_3']} {cx:.6f} {cy:.6f} {bw/W:.6f} {bh/H:.6f}")
                n_box += 1
            with open(f"{args.out}/labels/{sp}/{stem}.txt", "w") as f:
                f.write("\n".join(lines))

    yaml_path = f"{args.out}/dentex.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for k in sorted(CLASS_NAMES):
            f.write(f"  {k}: {CLASS_NAMES[k]}\n")
    print(f"OK: {n_box} boxes written. Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--out", default="/content/yolo")  # local Colab = fast
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    convert(ap.parse_args())
