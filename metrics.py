"""
metrics.py — Metrics comparing enhanced output to low-light input.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List


def psnr(enhanced: torch.Tensor, low_light: torch.Tensor, max_val: float = 1.0) -> float:
    if enhanced.dim() == 3:
        enhanced  = enhanced.unsqueeze(0)
        low_light = low_light.unsqueeze(0)
    mse = F.mse_loss(enhanced, low_light, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1).clamp(min=1e-10)
    return (10.0 * torch.log10(max_val ** 2 / mse)).mean().item()


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim(enhanced: torch.Tensor, low_light: torch.Tensor, max_val: float = 1.0,
         win_size: int = 11, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> float:
    if enhanced.dim() == 3:
        enhanced  = enhanced.unsqueeze(0)
        low_light = low_light.unsqueeze(0)

    B, C, H, W = enhanced.shape
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    g      = _gaussian_kernel(win_size, sigma).to(enhanced.device)
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).expand(C, 1, win_size, win_size)
    pad    = win_size // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1, mu2       = conv(enhanced), conv(low_light)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu12           = mu1 * mu2

    s1  = conv(enhanced  * enhanced)  - mu1_sq
    s2  = conv(low_light * low_light) - mu2_sq
    s12 = conv(enhanced  * low_light) - mu12

    num = (2 * mu12 + c1) * (2 * s12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)
    return (num / (den + 1e-8)).mean().item()


def mabd(frames: List[torch.Tensor]) -> float:
    if len(frames) < 2:
        return 0.0
    brightnesses = []
    for f in frames:
        if f.dim() == 4:
            f = f.squeeze(0)
        brightnesses.append(
            (0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]).mean().item()
        )
    diffs = [abs(brightnesses[i] - brightnesses[i - 1]) for i in range(1, len(brightnesses))]
    return float(np.mean(diffs))


def spaq(frame: torch.Tensor) -> float:
    """Stub — returns -1.0 until real SPAQ model is integrated."""
    return -1.0


def evaluate_sequence(enhanced_frames: List[torch.Tensor],
                      low_light_frames: List[torch.Tensor]) -> dict:
    psnr_vals, ssim_vals = [], []
    for e, l in zip(enhanced_frames, low_light_frames):
        psnr_vals.append(psnr(e, l))
        ssim_vals.append(ssim(e, l))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "mabd": mabd(enhanced_frames),
        "spaq": spaq(enhanced_frames[0]) if enhanced_frames else -1.0,
    }