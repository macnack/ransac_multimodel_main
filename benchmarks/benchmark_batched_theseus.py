"""
Batched throughput benchmark: scipy (Python loop) vs theseus (single batched call).

Picks one real sample from `tensors/` (default sample 128, N=196), replicates
its correspondences B times along the batch dimension, and runs:
  - scipy: B sequential `optimize_homography(...)` calls,
  - theseus: a single `refine_homography_theseus_torch(...)` call with shape
    (B, N, 2) inputs.

This is the regime where theseus *should* shine on GPU: amortizes per-call
launch overhead and uses cuSOLVER on a (B, 4, 4) batched dense system.

Run from any CWD:
    PYTHONPATH=<repo> python -m benchmarks.benchmark_batched_theseus \
        --batch-sizes 1,4,16,64 --device cpu
    PYTHONPATH=<repo> python -m benchmarks.benchmark_batched_theseus \
        --batch-sizes 1,4,16,64 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.benchmark_numpy_vs_torch import load_case_from_tensors  # noqa: E402
from ransac_multimodel.homography import optimize_homography  # noqa: E402
from ransac_multimodel.homography_theseus import (  # noqa: E402
    refine_homography_theseus_torch,
)
from ransac_multimodel.homography_torch_lm import (  # noqa: E402
    refine_homography_torch_lm_torch,
)


def _maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, repeats: int, warmup: int, device: str = "cpu") -> List[float]:
    for _ in range(warmup):
        fn()
        _maybe_sync(device)
    out = []
    for _ in range(repeats):
        _maybe_sync(device)
        t0 = time.perf_counter()
        fn()
        _maybe_sync(device)
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1000.0)
    return out


def _ransac_init_np(pts_A, peaks_B, means_B, use_means: bool = False):
    pts = means_B if use_means else (peaks_B if peaks_B is not None else means_B)
    H, _ = cv2.findHomography(
        pts_A, pts, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.995
    )
    if H is None:
        H = np.eye(3)
    return H / H[2, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        default=os.path.join(_REPO_ROOT, "tensors"),
        help="Path to .pt tensor dir; defaults to <repo>/tensors so the script "
             "works regardless of CWD.",
    )
    ap.add_argument("--sample-id", type=int, default=128, help="real sample whose correspondences to replicate")
    ap.add_argument("--batch-sizes", default="1,4,16,64")
    ap.add_argument("--model", choices=["sRT", "full"], default="sRT")
    ap.add_argument("--device", default="cpu", help="cpu or cuda — applies to theseus only")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-iter-theseus", type=int, default=100)
    ap.add_argument("--step-size-theseus", type=float, default=0.1)
    ap.add_argument("--output", default="benchmarks/results/batched_scipy_vs_theseus.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]

    case = load_case_from_tensors(args.sample_id, args.dataset_dir, quiet=True)
    N = case.pts_A.shape[0]
    print(f"sample {args.sample_id}: N={N}, model={args.model}, device={args.device}")

    # Pre-compute the per-element RANSAC init once — RANSAC is part of every
    # production pipeline but it isn't what we're comparing. Both methods are
    # given the same H_init so the timing isolates the LM refinement.
    H_init_np = _ransac_init_np(case.pts_A, case.peaks_B, case.means_B)

    rows: List[Dict[str, Any]] = []

    print(f"\n{'B':>4}  {'scipy ms':>10}  {'theseus ms':>10}  {'torchLM ms':>10}  {'sc/it ms':>9}  {'th/it ms':>9}  {'tlm/it ms':>10}  {'thSpdUp':>8}  {'tlmSpdUp':>9}")
    print("-" * 110)

    for B in sizes:
        # --- scipy: B sequential calls ---
        def scipy_call():
            for _ in range(B):
                optimize_homography(
                    case.pts_A,
                    case.means_B,
                    case.covs_B,
                    peaks_B=case.peaks_B,
                    model=args.model,
                    quiet=True,
                )

        t_scipy = _time(scipy_call, repeats=args.repeats, warmup=args.warmup, device="cpu")

        # --- theseus: one batched call ---
        device = args.device
        dtype = torch.float64
        pts_A_b = torch.from_numpy(case.pts_A.astype(np.float64)).to(device=device, dtype=dtype)
        means_B_b = torch.from_numpy(case.means_B.astype(np.float64)).to(device=device, dtype=dtype)
        covs_B_b = torch.from_numpy(case.covs_B.astype(np.float64)).to(device=device, dtype=dtype)
        H_init_b = torch.from_numpy(H_init_np).to(device=device, dtype=dtype)

        # Replicate B times along leading dim — same correspondences, just stacked.
        pts_A_batch = pts_A_b.unsqueeze(0).expand(B, -1, -1).contiguous()
        means_B_batch = means_B_b.unsqueeze(0).expand(B, -1, -1).contiguous()
        covs_B_batch = covs_B_b.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        H_init_batch = H_init_b.unsqueeze(0).expand(B, -1, -1).contiguous()

        def theseus_call():
            with torch.no_grad():
                refine_homography_theseus_torch(
                    pts_A_batch,
                    means_B_batch,
                    covs_B_batch,
                    H_init_batch,
                    model=args.model,
                    max_iter=args.max_iter_theseus,
                    step_size=args.step_size_theseus,
                )

        t_th = _time(theseus_call, repeats=args.repeats, warmup=args.warmup, device=device)

        # --- pure-torch LM: single batched call ---
        def torch_lm_call():
            with torch.no_grad():
                refine_homography_torch_lm_torch(
                    pts_A_batch,
                    means_B_batch,
                    covs_B_batch,
                    H_init_batch,
                    model=args.model,
                    max_iter=args.max_iter_theseus,
                )

        t_tlm = _time(torch_lm_call, repeats=args.repeats, warmup=args.warmup, device=device)

        scipy_med = float(statistics.median(t_scipy))
        th_med = float(statistics.median(t_th))
        tlm_med = float(statistics.median(t_tlm))
        per_item_th = th_med / B
        per_item_sc = scipy_med / B
        per_item_tlm = tlm_med / B
        th_speedup = scipy_med / th_med if th_med > 0 else float("inf")
        tlm_speedup = scipy_med / tlm_med if tlm_med > 0 else float("inf")

        print(
            f"{B:>4}  {scipy_med:>10.2f}  {th_med:>10.2f}  {tlm_med:>10.2f}  "
            f"{per_item_sc:>9.2f}  {per_item_th:>9.2f}  {per_item_tlm:>10.2f}  "
            f"{th_speedup:>7.2f}x  {tlm_speedup:>8.2f}x"
        )

        rows.append({
            "batch_size": B,
            "n_points": int(N),
            "model": args.model,
            "device": args.device,
            "scipy_loop_ms_median": scipy_med,
            "scipy_loop_ms_p95": float(np.percentile(t_scipy, 95)),
            "theseus_batch_ms_median": th_med,
            "theseus_batch_ms_p95": float(np.percentile(t_th, 95)),
            "torch_lm_batch_ms_median": tlm_med,
            "torch_lm_batch_ms_p95": float(np.percentile(t_tlm, 95)),
            "scipy_per_item_ms": per_item_sc,
            "theseus_per_item_ms": per_item_th,
            "torch_lm_per_item_ms": per_item_tlm,
            "speedup_theseus_over_scipy": th_speedup,
            "speedup_torch_lm_over_scipy": tlm_speedup,
        })

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sample_id": args.sample_id, "rows": rows}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
