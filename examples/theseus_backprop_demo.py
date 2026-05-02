"""
Backprop-through-refinement demo.

Sets up the synthetic correspondence case, declares the per-correspondence
covariances as a learnable torch parameter (treating them as if they came
from an upstream feature head), runs the Theseus LM refinement, computes a
geometry loss against the ground-truth homography, and backprops to update
the covariances.

Run from any CWD:
    PYTHONPATH=<repo> python -m examples.theseus_backprop_demo
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case  # noqa: E402
from ransac_multimodel.homography_theseus import (  # noqa: E402
    refine_homography_theseus_torch,
)


def four_corner_loss(H_pred: torch.Tensor, H_gt: torch.Tensor, w: float, h: float) -> torch.Tensor:
    """Mean L2 between the 4 image corners projected by H_pred vs H_gt."""
    corners = torch.tensor(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
        dtype=H_pred.dtype,
        device=H_pred.device,
    )
    ones = torch.ones(corners.shape[0], 1, dtype=H_pred.dtype, device=H_pred.device)
    pts = torch.cat([corners, ones], dim=-1)
    p_pred = (H_pred @ pts.T).T
    p_gt = (H_gt @ pts.T).T
    p_pred = p_pred[:, :2] / (p_pred[:, 2:3] + 1e-6)
    p_gt = p_gt[:, :2] / (p_gt[:, 2:3] + 1e-6)
    return torch.linalg.norm(p_pred - p_gt, dim=-1).mean()


def main():
    case = build_synthetic_case(seed=1234)
    n = case.pts_A.shape[0]

    pts_A = torch.as_tensor(case.pts_A, dtype=torch.float64)
    means_B = torch.as_tensor(case.means_B, dtype=torch.float64)

    # Make per-correspondence covariance scales a learnable parameter.
    # Start from the synthetic case's identity-ish covariance and let the
    # outer loss reshape them. We parametrize as log_scale so optimization
    # stays positive-definite.
    base_cov = torch.as_tensor(case.covs_B, dtype=torch.float64)
    log_scale = torch.zeros(n, dtype=torch.float64, requires_grad=True)

    H_gt = torch.as_tensor(case.H_gt, dtype=torch.float64)
    H_init_np = np.eye(3, dtype=np.float64)
    H_init = torch.as_tensor(H_init_np, dtype=torch.float64)

    outer_optim = torch.optim.Adam([log_scale], lr=0.1)

    print(f"{'iter':>4}  {'corner_err_px':>14}  {'log_scale.mean':>15}  {'log_scale.std':>14}")
    for it in range(20):
        outer_optim.zero_grad(set_to_none=True)

        # Build covariances for this iteration: scale * base_cov.
        scale = torch.exp(log_scale).reshape(n, 1, 1)
        covs = scale * base_cov

        H_opt = refine_homography_theseus_torch(
            pts_A,
            means_B,
            covs,
            H_init,
            model="sRT",
            f_scale=2.0,
            max_iter=30,
            step_size=1.0,
            backward_mode="implicit",
        )

        loss = four_corner_loss(H_opt[0], H_gt, w=1024.0, h=1024.0)
        loss.backward()
        outer_optim.step()

        print(
            f"{it:>4}  {float(loss):>14.5f}  {float(log_scale.mean()):>15.4f}  {float(log_scale.std()):>14.4f}"
        )

    print("\nFinal H_opt:")
    print(np.round(H_opt[0].detach().cpu().numpy(), 4))
    print("H_gt:")
    print(np.round(H_gt.cpu().numpy(), 4))


if __name__ == "__main__":
    main()
