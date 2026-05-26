"""
scripts/inpaint_fashionmnist.py  —  Inverse problem: pixel inpainting on FashionMNIST
=======================================================================================

Loads a trained VP score model and reconstructs FashionMNIST test images
that have been corrupted by a random pixel mask (75% of pixels zeroed).

The reverse-diffusion sampler is modified (see VPSDE.inpaint) to condition
on the observed measurements at every step, following the projection-based
approach of Song et al. (2022), "Solving Inverse Problems in Medical Imaging
with Score-Based Generative Models" (Algorithm 1).

Usage::
    python scripts/inpaint_fashionmnist.py \\
        --checkpoint runs/vp/best.pt \\
        --n_images 8 --corruption 0.75 --out runs/inpaint.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from diffusion.unet import UNet
from diffusion.vp import VPSDE


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained VP model .pt file.")
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--out",        type=str, default="runs/inpaint.png")
    p.add_argument("--n_images",   type=int, default=8)
    p.add_argument("--corruption", type=float, default=0.75,
                   help="Fraction of pixels to zero out (per image).")
    p.add_argument("--beta_min",   type=float, default=0.01)
    p.add_argument("--beta_max",   type=float, default=5.0)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--num_steps",  type=int,   default=1000)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_test_images(data_dir: str, n: int, seed: int) -> torch.Tensor:
    """Return n FashionMNIST test images normalised to [-1, 1], shape (n,1,28,28)."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    imgs = torch.stack([ds[int(i)][0] for i in idx], dim=0)
    return imgs


def make_pixel_mask(shape: tuple[int, int, int, int],
                    corruption: float,
                    device: torch.device,
                    generator: torch.Generator) -> torch.Tensor:
    """Random per-image pixel mask. 1 = observed, 0 = missing. Shape (B,1,H,W)."""
    B, _, H, W = shape
    p_keep = 1.0 - corruption
    m = (torch.rand((B, 1, H, W), device=device, generator=generator) < p_keep).float()
    return m


def psnr(reconstructed: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR (dB) for images in [-1, 1]. Returns shape (B,)."""
    # Convert to [0, 1] for a standard MAX=1 PSNR.
    rec01   = (reconstructed.clamp(-1, 1) + 1) / 2
    clean01 = (clean.clamp(-1, 1) + 1) / 2
    mse = ((rec01 - clean01) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def plot_inpainting(clean: torch.Tensor,
                    corrupted: torch.Tensor,
                    reconstructed: torch.Tensor,
                    psnrs: torch.Tensor,
                    out_path: str,
                    corruption: float) -> None:
    """3-row figure: clean | corrupted | reconstructed (+ PSNR title per col)."""
    n = clean.size(0)

    def to_img(x: torch.Tensor) -> np.ndarray:
        return ((x.clamp(-1, 1) + 1) / 2).squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 4.5))
    if n == 1:
        axes = axes[:, None]

    for j in range(n):
        axes[0, j].imshow(to_img(clean[j]), cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        axes[1, j].imshow(to_img(corrupted[j]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].axis("off")
        axes[2, j].imshow(to_img(reconstructed[j]), cmap="gray", vmin=0, vmax=1)
        axes[2, j].set_title(f"PSNR\n{psnrs[j].item():.2f} dB", fontsize=8)
        axes[2, j].axis("off")

    axes[0, 0].set_ylabel("Clean",         fontsize=10)
    axes[1, 0].set_ylabel(f"Corrupted ({int(corruption*100)}%)", fontsize=10)
    axes[2, 0].set_ylabel("Reconstructed", fontsize=10)
    # Re-enable y-axis label visibility (set_axis_off above hid it); use text instead.
    for ax, label in zip(axes[:, 0], ["Clean",
                                      f"Corrupted ({int(corruption*100)}%)",
                                      "Reconstructed"]):
        ax.text(-0.15, 0.5, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=10, rotation=90)

    fig.suptitle(
        f"VP score-based inpainting on FashionMNIST  "
        f"(mean PSNR = {psnrs.mean().item():.2f} dB)", fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 1) Load model + SDE
    sde   = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded VP checkpoint: {args.checkpoint}")

    # 2) Grab clean test images
    clean = load_test_images(args.data_dir, args.n_images, args.seed).to(device)

    # 3) Build measurement matrix A as a random pixel mask
    gen = torch.Generator(device=device).manual_seed(args.seed)
    mask = make_pixel_mask(clean.shape, args.corruption, device, gen)
    corrupted = clean * mask  # A x: missing pixels -> 0

    # 4) Conditional reverse-diffusion reconstruction
    print(f"Running inpaint sampler ({args.num_steps} steps) ...")
    reconstructed = sde.inpaint(
        model, corrupted, mask,
        num_steps=args.num_steps, device=device,
    )

    # Keep observed pixels exact (we know them perfectly).
    reconstructed = mask * clean + (1.0 - mask) * reconstructed

    # 5) PSNR + plot
    psnrs = psnr(reconstructed, clean)
    for i, p in enumerate(psnrs):
        print(f"  image {i}: PSNR = {p.item():.2f} dB")
    print(f"Mean PSNR: {psnrs.mean().item():.2f} dB")

    plot_inpainting(clean, corrupted, reconstructed, psnrs, args.out, args.corruption)


if __name__ == "__main__":
    main()
