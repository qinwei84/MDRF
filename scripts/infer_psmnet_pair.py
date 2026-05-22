#!/usr/bin/env python3
"""Run single-pair inference with the PSMNet host baseline used for MDRF transfer tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psmnet import PSMNet  # noqa: E402


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    std = float(image.std())
    return (image - float(image.mean())) / (std if std > 1e-6 else 1.0)


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image[None, None]).to(device=device, dtype=torch.float32)


def clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or contain a state_dict/model_state_dict field.")
    return {str(k).removeprefix("module."): v for k, v in checkpoint.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-disp", default=-128, type=int)
    parser.add_argument("--max-disp", default=64, type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--non-strict", action="store_true", help="Allow missing or unexpected checkpoint keys.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    left_np = read_gray(args.left)
    right_np = read_gray(args.right)
    if left_np.shape != right_np.shape:
        raise ValueError(f"Left/right image size mismatch: {left_np.shape} vs {right_np.shape}")

    model = PSMNet(maxdisp=args.max_disp, min_disp=args.min_disp, in_channels=1)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint), strict=not args.non_strict)
    model.to(device).eval()

    with torch.no_grad():
        prediction = model(to_tensor(left_np, device), to_tensor(right_np, device))

    disparity = prediction.squeeze().detach().cpu().numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), disparity)
    print(f"Saved disparity to {args.output}")


if __name__ == "__main__":
    main()
