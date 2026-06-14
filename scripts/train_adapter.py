"""
Training loop Medical SAM Adapter di DENTEX lesion (Stage 2).

Recipe (Wu et al., 2025): box prompt -> mask. Loss = Dice + BCE.
Checkpoint adapter (kecil, ~20-50MB) DISIMPAN KE DRIVE supaya aman dari
reset Colab. Jalankan via notebooks/01_sam_adapter_train.ipynb.

Catatan compute: SAM ViT-H image encoder berat -> batch kecil (1-2) di T4,
gunakan gradient accumulation. Per-kelas Dice dilaporkan (penting untuk H4
periapical n=158).
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dentex_dataset import (
    CLASS_NAMES,
    DentexLesionDataset,
    load_records,
    stratified_split,
)
from sam_adapter import (
    adapter_state_dict,
    inject_adapters,
    trainable_report,
)
from segment_anything import sam_model_registry


def dice_bce_loss(logits, gt):
    """logits, gt: B x 1 x H x W. gt in {0,1}."""
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    p = torch.sigmoid(logits)
    num = 2 * (p * gt).sum(dim=(1, 2, 3)) + 1
    den = p.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) + 1
    dice = 1 - (num / den).mean()
    return bce + dice


@torch.no_grad()
def dice_score(logits, gt, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    num = 2 * (p * gt).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) + 1e-6
    return (num / den)  # B


def run_image_encoder(sam, batch, device):
    """SAM image encoder (adapter di dalam -> dapat gradient). Output feats.
    image_1024 sudah ter-normalisasi + pad ke 1024x1024 di dataset."""
    return sam.image_encoder(batch["image_1024"].to(device))


def decode_masks(sam, feats, boxes, device):
    sparse, dense = sam.prompt_encoder(points=None, boxes=boxes.to(device), masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=feats,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    # upsample low-res (256) -> 1024
    return F.interpolate(low_res, (1024, 1024), mode="bilinear", align_corners=False)


def train(args):
    device = "cuda"
    js = glob.glob(f"{args.drive}/data/dentex/**/*disease*.json", recursive=True)[0]
    xr = os.path.join(os.path.dirname(js), "xrays")
    recs = load_records(js, xr)
    tr_rec, va_rec = stratified_split(recs, val_frac=args.val_frac)
    print(f"Lesion: {len(recs)} | train {len(tr_rec)} | val {len(va_rec)}")

    tr = DataLoader(DentexLesionDataset(tr_rec), batch_size=args.bs, shuffle=True,
                    num_workers=2, collate_fn=lambda b: b)
    va = DataLoader(DentexLesionDataset(va_rec), batch_size=args.bs, shuffle=False,
                    num_workers=2, collate_fn=lambda b: b)

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt)
    inject_adapters(sam)      # tambah adapter SEBELUM pindah device
    sam.to(device)            # pindahkan semua (base + adapter) ke GPU
    trainable_report(sam)

    opt = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    os.makedirs(f"{args.drive}/checkpoints", exist_ok=True)
    best = 0.0

    for ep in range(args.epochs):
        sam.train()
        run_loss, steps = 0.0, 0
        opt.zero_grad()
        for bi, samples in enumerate(tr):
            # collate manual (batch=list of dict) — proses per-sample lalu stack
            boxes = torch.stack([s["box"] for s in samples])[:, None, :]  # B x 1 x 4
            gts = torch.stack([s["gt_mask"] for s in samples])[:, None].to(device)
            batch = {"image_1024": torch.stack([s["image_1024"] for s in samples])}

            feats = run_image_encoder(sam, batch, device)
            logits = decode_masks(sam, feats, boxes, device)
            loss = dice_bce_loss(logits, gts) / args.accum
            loss.backward()
            run_loss += loss.item() * args.accum
            steps += 1
            if (bi + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
            if bi % 50 == 0:
                print(f"  ep{ep} step{bi}/{len(tr)} loss {run_loss/steps:.4f}")

        # ---- validasi: Dice global + per-kelas ----
        sam.eval()
        per_cls = defaultdict(list)
        with torch.no_grad():
            for samples in va:
                boxes = torch.stack([s["box"] for s in samples])[:, None, :]
                gts = torch.stack([s["gt_mask"] for s in samples])[:, None].to(device)
                batch = {"image_1024": torch.stack([s["image_1024"] for s in samples])}
                feats = run_image_encoder(sam, batch, device)
                logits = decode_masks(sam, feats, boxes, device)
                ds = dice_score(logits, gts).cpu().numpy()
                for s, d in zip(samples, ds):
                    per_cls[s["cls"]].append(d)
        mdice = np.mean([d for v in per_cls.values() for d in v])
        print(f"[ep{ep}] train_loss {run_loss/steps:.4f} | val Dice {mdice:.4f}")
        for c in sorted(per_cls):
            print(f"    {CLASS_NAMES[c]:20s} Dice {np.mean(per_cls[c]):.4f} (n={len(per_cls[c])})")

        # ---- checkpoint terbaik -> Drive ----
        if mdice > best:
            best = mdice
            path = f"{args.drive}/checkpoints/adapter_best.pth"
            torch.save({"state": adapter_state_dict(sam), "dice": best, "epoch": ep}, path)
            print(f"    ✅ saved {path} (Dice {best:.4f})")

    print(f"Done. Best val Dice {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--sam_ckpt", default="/content/drive/MyDrive/opg-live/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)  # effective batch 8
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    train(ap.parse_args())
