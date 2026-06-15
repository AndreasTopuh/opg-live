"""
DENTEX lesion dataset for training the Medical SAM Adapter (Stage 2).

Per-sample output:
  image      : uint8 HxWx3 (RGB, original) — resized to 1024 by the SAM transform
  box        : lesion bbox [x0,y0,x1,y1] in 1024 coordinates (SAM prompt)
  gt_mask    : 1024x1024 float {0,1} — lesion mask from polygon
  cls        : int 0-3 (category_id_3) — Impacted/Caries/Periapical/Deep Caries

DENTEX disease schema (HIERARCHICAL):
  annotations[i] = {image_id, bbox[x,y,w,h], segmentation[poly], category_id_3, ...}
  categories_3   = {0:Impacted, 1:Caries, 2:Periapical Lesion, 3:Deep Caries}

1 sample = 1 lesion (not 1 image). Stratified per-class split so that
Periapical (n=158) stays represented in both train and val.
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

# SAM normalisation constants (same for ViT-B/L/H)
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def poly_to_mask(segmentation, h, w):
    """COCO polygon -> binary mask HxW."""
    rles = mask_utils.frPyObjects(segmentation, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.uint8)


def stratified_split(records, val_frac=0.15, seed=42):
    """Split records by class so each class is proportionally represented."""
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
    """Parse the DENTEX disease json -> list of per-lesion records."""
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
                "ann_id": a["id"],
                "image_id": a["image_id"],
                "img_file": id2file[a["image_id"]],
                "img_path": os.path.join(xrays_dir, id2file[a["image_id"]]),
                "bbox_xywh": a["bbox"],
                "segmentation": a["segmentation"],
                "cls": a["category_id_3"],
                "h": h,
                "w": w,
            }
        )
    return records


def sample_per_class(records, n_per_class, seed=42):
    """Take n_per_class lesions per class (stratified). Used to build a
    cost-controlled GPT-4o eval set. Deterministic (seed)."""
    by_cls = defaultdict(list)
    for r in records:
        by_cls[r["cls"]].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for cls, items in by_cls.items():
        idx = rng.permutation(len(items))[:n_per_class]
        out += [items[i] for i in idx]
    return out


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

        # GT mask from polygon
        mask = poly_to_mask(r["segmentation"], h, w)

        # bbox xywh -> xyxy
        x, y, bw, bh = r["bbox_xywh"]
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)

        # Resize into the 1024 space (SAM ResizeLongestSide)
        img_1024 = self.transform.apply_image(image)  # H'xW'x3, longest side=1024
        box_1024 = self.transform.apply_boxes(box[None, :], (h, w))[0]
        nh, nw = self.transform.get_preprocess_shape(h, w, self.img_size)
        mask_1024 = np.array(
            Image.fromarray(mask).resize((nw, nh), Image.NEAREST)
        )

        # SAM normalisation then PAD to 1024x1024 (right & bottom with 0).
        # Padding here makes all samples the same size -> can be stacked into a batch.
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
    # Smoke test (run on Colab): check count & one sample
    import glob

    DRIVE = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/opg-live")
    js = glob.glob(f"{DRIVE}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    recs = load_records(js, xr)
    print("Total lesion records:", len(recs))  # should be 3529
    tr, va = stratified_split(recs)
    print("Train / Val:", len(tr), len(va))
    ds = DentexLesionDataset(tr)
    s = ds[0]
    print("image_1024:", s["image_1024"].shape, "| box:", s["box"].tolist())
    print("gt_mask:", s["gt_mask"].shape, "sum:", s["gt_mask"].sum().item(), "| cls:", CLASS_NAMES[s["cls"]])
