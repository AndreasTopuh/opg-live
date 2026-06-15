"""
Konversi DENTEX enumeration (quadrant + tooth) -> format YOLOv8 32-kelas FDI.

Untuk detektor GIGI (Stage 1 FDI): tiap gigi punya FDI 2-digit = kuadran+nomor.
  category_id_1 (kuadran, id 0-3 -> nama 1-4)
  category_id_2 (gigi,    id 0-7 -> nama 1-8)
  class YOLO = cat1_id*8 + cat2_id  (0-31)
  FDI name   = f"{kuadran}{gigi}"   (11..18, 21..28, 31..38, 41..48)

Detektor ini dipakai untuk assign FDI ke deteksi penyakit (lewat overlap),
sehingga Stage 1 = bbox + FDI + diagnosis (prediksi penuh, tanpa GT).

Output: /content/yolo_enum/{images,labels}/{train,val} + dentex_enum.yaml
"""
import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np


def fdi_of(cat1_name, cat2_name):
    return f"{cat1_name}{cat2_name}"


def convert(args):
    js = glob.glob(f"{args.drive}/data/dentex/**/*enumeration.json", recursive=True)
    js = [p for p in js if "disease" not in p][0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    d = json.load(open(js))

    c1 = {c["id"]: c["name"] for c in d["categories_1"]}   # kuadran
    c2 = {c["id"]: c["name"] for c in d["categories_2"]}   # gigi
    # class id 0-31 -> FDI name
    names = {}
    for q in sorted(c1):
        for t in sorted(c2):
            names[q * 8 + t] = fdi_of(c1[q], c2[t])

    images = {im["id"]: im for im in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    img_ids = sorted(anns_by_img.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(img_ids)
    n_val = int(len(img_ids) * args.val_frac)
    splits = {"val": set(img_ids[:n_val]), "train": set(img_ids[n_val:])}
    print(f"Gambar: {len(img_ids)} | train {len(splits['train'])} | val {len(splits['val'])}")

    for sp in ["train", "val"]:
        os.makedirs(f"{args.out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{sp}", exist_ok=True)

    n_box = 0
    for sp, ids in splits.items():
        for iid in ids:
            im = images[iid]
            W, H = im["width"], im["height"]
            src = os.path.join(xr, im["file_name"])
            if not os.path.exists(src):
                continue
            shutil.copy(src, f"{args.out}/images/{sp}/{im['file_name']}")
            stem = os.path.splitext(im["file_name"])[0]
            lines = []
            for a in anns_by_img[iid]:
                cls = a["category_id_1"] * 8 + a["category_id_2"]
                x, y, bw, bh = a["bbox"]
                cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw/W:.6f} {bh/H:.6f}")
                n_box += 1
            with open(f"{args.out}/labels/{sp}/{stem}.txt", "w") as f:
                f.write("\n".join(lines))

    with open(f"{args.out}/dentex_enum.yaml", "w") as f:
        f.write(f"path: {args.out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for k in sorted(names):
            f.write(f"  {k}: '{names[k]}'\n")
    print(f"✅ {n_box} gigi ditulis, 32 kelas FDI. YAML: {args.out}/dentex_enum.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--out", default="/content/yolo_enum")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    convert(ap.parse_args())
