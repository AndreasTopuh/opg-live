"""
Medical SAM Adapter (Wu et al., 2025, Medical Image Analysis) — lightweight.

Strategy: SAM ViT-H is FROZEN. Small Adapter modules (bottleneck down->act->up)
are inserted into every image-encoder transformer block. Only the adapters +
mask decoder are trainable (~2-3% params). No full fine-tune, no SAM 2.

NOTE: this follows the MSA adapter pattern (two adapters per block: one after
attention, one parallel to the MLP). Validate Dice on the val set before
production use; for exact fidelity, swap in the official Medical-SAM-Adapter
(MSA) repo. Adapters are initialised near identity (up=0) so training starts
from the original SAM behaviour.
"""
import types

import torch
import torch.nn as nn
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)


class Adapter(nn.Module):
    """Bottleneck adapter: x + up(act(down(x))). Init up=0 -> identity at start."""

    def __init__(self, dim, mlp_ratio=0.25, act=nn.GELU, skip=True):
        super().__init__()
        hidden = max(1, int(dim * mlp_ratio))
        self.skip = skip
        self.down = nn.Linear(dim, hidden)
        self.act = act()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.up(self.act(self.down(x)))
        return x + h if self.skip else h


def _patched_block_forward(self, x):
    """Forward ViT Block + adapter. Mirrors segment_anything Block.forward."""
    shortcut = x
    x = self.norm1(x)
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)
    x = self.attn(x)
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
    x = self.adapter_attn(x)                     # adapter after attention
    x = shortcut + x
    xn = self.norm2(x)
    x = x + self.mlp(xn) + self.scale * self.adapter_mlp(xn)  # adapter parallel to MLP
    return x


def inject_adapters(sam, mlp_ratio=0.25, scale=0.5):
    """Freeze SAM, add an adapter to each image-encoder block. The mask decoder
    is left trainable. Returns sam (in-place)."""
    # 1. Freeze everything
    for p in sam.parameters():
        p.requires_grad = False

    # 2. Insert adapters into each block + patch forward
    for blk in sam.image_encoder.blocks:
        dim = blk.norm1.weight.shape[0]
        blk.adapter_attn = Adapter(dim, mlp_ratio)
        blk.adapter_mlp = Adapter(dim, mlp_ratio, skip=False)
        blk.scale = scale
        blk.forward = types.MethodType(_patched_block_forward, blk)

    # 3. Make adapters trainable
    for blk in sam.image_encoder.blocks:
        for p in blk.adapter_attn.parameters():
            p.requires_grad = True
        for p in blk.adapter_mlp.parameters():
            p.requires_grad = True

    # 4. Mask decoder trainable (prompt encoder stays frozen)
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True

    return sam


def trainable_report(sam):
    total = sum(p.numel() for p in sam.parameters())
    train = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    print(f"Trainable: {train/1e6:.2f}M / {total/1e6:.0f}M  ({100*train/total:.2f}%)")
    return train, total


def adapter_state_dict(sam):
    """Only the trainable weights (adapter + mask decoder) -> small checkpoint."""
    return {k: v.cpu() for k, v in sam.state_dict().items()
            if "adapter_" in k or "mask_decoder" in k}


def load_adapter_state(sam, state):
    sam.load_state_dict(state, strict=False)
    return sam
