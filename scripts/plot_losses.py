"""
scripts/plot_losses.py  —  Plot training/validation losses on a log scale
==========================================================================

Loads .npy loss arrays saved by scripts/train_vp.py or scripts/train_rectflow.py
and plots them with a log-scale y-axis.

Usage::
    # VP run (has both train and val):
    python scripts/plot_losses.py --run_dir runs/vp

    # Rectified Flow run (train only):
    python scripts/plot_losses.py --run_dir runs/rectflow

    # Or specify files explicitly:
    python scripts/plot_losses.py --train runs/vp/train_losses.npy \\
                                  --val   runs/vp/val_losses.npy \\
                                  --out   runs/vp/losses.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None,
                   help="Directory containing train_losses.npy (and optionally val_losses.npy).")
    p.add_argument("--train",   type=str, default=None, help="Path to train_losses.npy.")
    p.add_argument("--val",     type=str, default=None, help="Path to val_losses.npy.")
    p.add_argument("--out",     type=str, default=None, help="Output figure path (default: <run_dir>/losses.png).")
    p.add_argument("--title",   type=str, default="Training loss (log scale)")
    p.add_argument("--show",    action="store_true", help="Display the plot interactively.")
    return p.parse_args()


def main():
    args = get_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        train_path = Path(args.train) if args.train else run_dir / "train_losses.npy"
        val_path   = Path(args.val)   if args.val   else run_dir / "val_losses.npy"
        out_path   = Path(args.out)   if args.out   else run_dir / "losses.png"
    else:
        if args.train is None:
            raise ValueError("Provide either --run_dir or --train.")
        train_path = Path(args.train)
        val_path   = Path(args.val) if args.val else None
        out_path   = Path(args.out) if args.out else train_path.with_name("losses.png")

    if not train_path.exists():
        raise FileNotFoundError(f"Train losses not found: {train_path}")
    train = np.load(train_path)
    epochs = np.arange(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train, label="train", marker="o", markersize=3)

    if val_path is not None and val_path.exists():
        val = np.load(val_path)
        ax.plot(np.arange(1, len(val) + 1), val, label="val", marker="s", markersize=3)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(args.title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
