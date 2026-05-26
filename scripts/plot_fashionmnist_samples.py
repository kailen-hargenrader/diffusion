"""
scripts/plot_fashionmnist_samples.py
=====================================
Plot 64 images from the FashionMNIST training set in an 8x8 grid and
print the class names and image dimensions.

Usage::
    python scripts/plot_fashionmnist_samples.py --out runs/fashionmnist_samples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out",      type=str, default="runs/fashionmnist_samples.png")
    p.add_argument("--n",        type=int, default=64)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--show",     action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    ds = datasets.FashionMNIST(
        args.data_dir, train=True, download=True, transform=transforms.ToTensor(),
    )

    # Dimensions
    img0, _ = ds[0]
    C, H, W = img0.shape
    print(f"Number of training images: {len(ds)}")
    print(f"Image dimensions (C, H, W): ({C}, {H}, {W})")
    print(f"Pixel value range: [{img0.min().item():.3f}, {img0.max().item():.3f}]")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print("Classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")

    # Sample n images
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=args.n, replace=False)

    side = int(np.ceil(np.sqrt(args.n)))
    fig, axes = plt.subplots(side, side, figsize=(side, side))
    for ax, i in zip(axes.flat, idx):
        img, label = ds[int(i)]
        ax.imshow(img.squeeze(0).numpy(), cmap="gray")
        ax.set_title(CLASS_NAMES[label], fontsize=5)
        ax.axis("off")
    for ax in axes.flat[len(idx):]:
        ax.axis("off")

    fig.suptitle(f"FashionMNIST samples ({args.n} images, {H}x{W})", fontsize=10)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
