"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def main():
    args = get_args()
    device = torch.device(args.device)

    # ----------------------------------------------------------------
    # 1) Prepare real reference images (FashionMNIST test set, [0,1])
    # ----------------------------------------------------------------
    work_root = tempfile.mkdtemp(prefix="kid_")
    real_dir = os.path.join(work_root, "real")
    os.makedirs(real_dir, exist_ok=True)

    tf = transforms.ToTensor()  # [0,1]
    real_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    n_real = min(args.n_samples, len(real_ds))
    print(f"Saving {n_real} real reference images to {real_dir} ...")
    for i in range(n_real):
        img, _ = real_ds[i]
        save_image(img, os.path.join(real_dir, f"{i:05d}.png"))

    # ----------------------------------------------------------------
    # 2) Load models
    # ----------------------------------------------------------------
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()
    print(f"Loaded VP checkpoint: {args.vp_checkpoint}")

    flow = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()
    print(f"Loaded RF checkpoint: {args.rf_checkpoint}")

    # ----------------------------------------------------------------
    # 3) DDIM-style deterministic VP sampler (probability-flow ODE):
    #    dx/dt = -½ β(t) x - ½ β(t) s_θ(x,t),  integrated from t=1 -> 0.
    # ----------------------------------------------------------------
    @torch.no_grad()
    def vp_ddim_sample(shape, num_steps):
        B = shape[0]
        dt = 1.0 / num_steps
        t1 = torch.ones(B, device=device)
        s1 = sde.sigma(t1).view(B, *([1] * (len(shape) - 1)))
        x = s1 * torch.randn(shape, device=device)
        for i in range(num_steps):
            t = 1.0 - i * dt
            tb = torch.full((B,), t, device=device)
            beta_t = sde.beta(tb).view(B, *([1] * (len(shape) - 1)))
            score = vp_model(x, tb)
            drift = -0.5 * beta_t * x - 0.5 * beta_t * score
            x = x - drift * dt
        return x

    # ----------------------------------------------------------------
    # 4) Sampling dispatch (batched to limit memory)
    # ----------------------------------------------------------------
    def generate(method: str, num_steps: int, n_samples: int, batch: int = 128) -> torch.Tensor:
        out = []
        remaining = n_samples
        while remaining > 0:
            bs = min(batch, remaining)
            shape = (bs, 1, 28, 28)
            if method == "em":
                x = sde.euler_maruyama(vp_model, shape, num_steps=num_steps, device=device)
            elif method == "ddim":
                x = vp_ddim_sample(shape, num_steps)
            elif method == "rectflow":
                x = flow.euler_sample(rf_model, shape, num_steps=num_steps, device=device)
            else:
                raise ValueError(method)
            out.append(x.cpu())
            remaining -= bs
        return torch.cat(out, dim=0)

    # ----------------------------------------------------------------
    # 5) Loop over (method, num_steps); compute KID via torch-fidelity
    # ----------------------------------------------------------------
    results: dict[tuple[str, int], tuple[float, float]] = {}
    for method in METHODS:
        for steps in STEP_COUNTS:
            tag = f"{method}_s{steps}"
            print(f"\n[{tag}] generating {args.n_samples} samples ...")
            samples = generate(method, steps, args.n_samples)
            gen_dir = os.path.join(work_root, tag)
            save_samples_to_dir(samples, gen_dir)
            metrics = compute_kid(gen_dir, real_dir)
            mean = float(metrics.get("kernel_inception_distance_mean", float("nan")))
            std  = float(metrics.get("kernel_inception_distance_std",  float("nan")))
            results[(method, steps)] = (mean, std)
            print(f"[{tag}] KID = {mean:.4f} ± {std:.4f}")

    # ----------------------------------------------------------------
    # 6) Print markdown table
    # ----------------------------------------------------------------
    header = "| method \\ steps | " + " | ".join(str(s) for s in STEP_COUNTS) + " |"
    sep    = "|" + "|".join(["---"] * (len(STEP_COUNTS) + 1)) + "|"
    print("\n\n### KID (mean ± std) — lower is better\n")
    print(header)
    print(sep)
    for method in METHODS:
        cells = []
        for steps in STEP_COUNTS:
            mean, std = results[(method, steps)]
            cells.append(f"{mean:.4f} ± {std:.4f}")
        print(f"| {method} | " + " | ".join(cells) + " |")

    print(f"\nArtifacts in: {work_root}")


if __name__ == "__main__":
    main()
