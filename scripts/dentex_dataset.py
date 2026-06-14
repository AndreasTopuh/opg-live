"""
DENTEX lesion dataset untuk training Medical SAM Adapter (Stage 2).

Output per sample:
  image      : uint8 HxWx3 (RGB, original) — di-resize ke 1024 oleh SAM transform
  box        : lesion bbox [x0,y0,x1,y1] dalam koordinat 1024 (prompt SAM)
  gt_mask    : 1024x1024 float {0,1} — lesion mask dari polygon
  cls        : int 0-3 (category_id_3) — Impacted/Caries/Periapical/Deep Caries

Schema DENTEX disease (HIERARKIS):
  annotations[i] = {image_id, bbox[x,y,w,h], segmentation[poly], category_id_3, ...}
  categories_3   = {0:Impacted, 1:Caries, 2:Periapical Lesion, 3:Deep Caries}

1 sample = 1 lesion (bukan 1 gambar). Stratified split per-kelas supaya
Periapical (n=158) tetap terwakili di train & val.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}

# Konstanta normalisasi SAM (sama untuk ViT-B/L/H)
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def poly_to_mask(segmentation, h, w):
    """COCO polygon -> binary mask HxW."""
    rles = mask_utils.frPyObjects(segmentation, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.uint8)


def stratified_split(records, val_frac=0.15, seed=42):
    """Split list of (img_path, ann) by class so tiap kelas proporsional."""
    by_cls = defaultdict(list)
    for r in records:
        by_cls[r["cls"]].append(r)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls, items in by_cls.items():
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_frac))
        val += [items[i] for i in idx[:n_val]]
        train += [items[i] for i in idx[n_val:]]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_records(disease_json, xrays_dir):
    """Parse DENTEX disease json -> list of per-lesion records."""
    d = json.load(open(disease_json))
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    id2hw = {im["id"]: (im["height"], im["width"]) for im in d["images"]}
    records = []
    for a in d["annotations"]:
        if not a.get("segmentation"):
            continue
        h, w = id2hw[a["image_id"]]
        records.append(
            {
                "img_path": os.path.join(xrays_dir, id2file[a["image_id"]]),
                "bbox_xywh": a["bbox"],
                "segmentation": a["segmentation"],
                "cls": a["category_id_3"],
                "h": h,
                "w": w,
            }
        )
    return records


class DentexLesionDataset(Dataset):
    def __init__(self, records, img_size=1024):
        self.records = records
        self.transform = ResizeLongestSide(img_size)
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        image = np.array(Image.open(r["img_path"]).convert("RGB"))
        h, w = r["h"], r["w"]

        # GT mask dari polygon
        mask = poly_to_mask(r["segmentation"], h, w)

        # bbox xywh -> xyxy
        x, y, bw, bh = r["bbox_xywh"]
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)

        # Resize ke ruang 1024 (SAM ResizeLongestSide)
        img_1024 = self.transform.apply_image(image)  # H'xW'x3, longest side=1024
        box_1024 = self.transform.apply_boxes(box[None, :], (h, w))[0]
        nh, nw = self.transform.get_preprocess_shape(h, w, self.img_size)
        mask_1024 = np.array(
            Image.fromarray(mask).resize((nw, nh), Image.NEAREST)
        )

        # Normalisasi SAM lalu PAD ke 1024x1024 (kanan & bawah dengan 0).
        # Pad di sini supaya semua sample sama ukuran -> bisa di-stack jadi batch.
        img_t = torch.as_tensor(img_1024, dtype=torch.float32).permute(2, 0, 1)  # 3xH'xW'
        img_t = (img_t - PIXEL_MEAN) / PIXEL_STD
        img_t = F.pad(img_t, (0, self.img_size - nw, 0, self.img_size - nh))     # 3x1024x1024

        mask_pad = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        mask_pad[:nh, :nw] = mask_1024

        return {
            "image_1024": img_t,                                       # 3x1024x1024 float (SAM-norm)
            "box": torch.as_tensor(box_1024, dtype=torch.float32),     # 4
            "gt_mask": torch.as_tensor(mask_pad, dtype=torch.float32), # 1024x1024
            "cls": int(r["cls"]),
            "orig_hw": (h, w),
        }


if __name__ == "__main__":
    # Smoke test (jalankan di Colab): cek jumlah & 1 sample
    import sys

    DRIVE = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/opg-live")
    import glob

    js = glob.glob(f"{DRIVE}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    recs = load_records(js, xr)
    print("Total lesion records:", len(recs))  # harus 3529
    tr, va = stratified_split(recs)
    print("Train / Val:", len(tr), len(va))
    ds = DentexLesionDataset(tr)
    s = ds[0]
    print("image_1024:", s["image_1024"].shape, "| box:", s["box"].tolist())
    print("gt_mask:", s["gt_mask"].shape, "sum:", s["gt_mask"].sum().item(), "| cls:", CLASS_NAMES[s["cls"]])
