"""End-to-end batched pipeline benchmark.

Measures the full inference pipeline (extraction -> RANSAC init -> refinement)
across four paths:

  1. ``np_seq``        - numpy ``find_gaussians`` + scipy ``optimize_homography``
                         in a Python loop (the reference baseline).
  2. ``torch_seq_cpu`` - torch ``find_gaussians_torch`` + torch-LM, both CPU,
                         in a Python loop.
  3. ``torch_seq_cuda``- same but CUDA.
  4. ``torch_batched`` - ``find_gaussians_torch_batch`` (one batched CUDA call)
                         + RANSAC loop + torch-LM. With ``--mode homogeneous``
                         (all frames have the same N) we batch the LM refine
                         too; with ``--mode heterogeneous`` we loop per-frame
                         on the LM step because varying N can't be batched
                         without padding+mask support.

Two batch composition modes:

* ``--mode homogeneous`` (default): replicate one sample B times. Same N
  everywhere, so the torch-LM batched call works. This is the upper-bound
  speedup for the batched path.
* ``--mode heterogeneous``: round-robin the 3 real samples (98, 122, 128) to
  fill the batch. Different N per frame, so the torch_batched path drops to
  per-frame LM. This is the realistic mixed-load scenario.

Run from any CWD::

    PYTHONPATH=<repo> python -m benchmarks.benchmark_e2e_batched \
        --batch-sizes 1,4,16,64 --mode homogeneous --device-cuda
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ransac_multimodel.correspondence import find_gaussians  # noqa: E402
from ransac_multimodel.correspondence_torch import (  # noqa: E402
    find_gaussians_torch,
    find_gaussians_torch_batch,
)
from ransac_multimodel.homography import optimize_homography  # noqa: E402
from ransac_multimodel.homography_torch_lm import (  # noqa: E402
    refine_homography_torch_lm_torch,
)


SAMPLE_IDS = [98, 122, 128]
DEFAULT_RANSAC = cv2.USAC_FAST


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn: Callable, repeats: int, warmup: int, device: str) -> List[float]:
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


def _ransac_init_one(pts_A: np.ndarray, peaks_B: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(
        pts_A, peaks_B, DEFAULT_RANSAC,
        ransacReprojThreshold=3.0, maxIters=5000, confidence=0.995,
    )
    if H is None:
        return np.eye(3, dtype=np.float64)
    return H / H[2, 2]


def _load_logits(dataset_dir: str, sid: int) -> torch.Tensor:
    path = os.path.join(dataset_dir, f"sample_{sid:03d}_tensor.pt")
    return torch.load(path, map_location="cpu")[16]["gm_cls"][0]


def _make_batch(
    dataset_dir: str, B: int, mode: str
) -> tuple[List[torch.Tensor], torch.Tensor]:
    """Return (list_of_per_frame_logits, stacked_(B,M,h,w)_tensor)."""
    if mode == "homogeneous":
        logits = _load_logits(dataset_dir, 128)  # smallest N, fastest
        items = [logits for _ in range(B)]
    elif mode == "heterogeneous":
        loaded = [_load_logits(dataset_dir, sid) for sid in SAMPLE_IDS]
        items = [loaded[i % len(loaded)] for i in range(B)]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    stacked = torch.stack(items, dim=0)
    return items, stacked


# --------------------------------------------------------------------------- #
# Pipeline implementations                                                    #
# --------------------------------------------------------------------------- #

def pipeline_np_seq(items: List[torch.Tensor]) -> List[np.ndarray]:
    """numpy extraction + scipy refinement, one frame at a time."""
    out = []
    for logits in items:
        pts_A, means_B, peaks_B, covs_B = find_gaussians(
            logits, adaptive_gauss_fit=False, log_missing_gaussians=False,
        )
        if pts_A.shape[0] < 4:
            out.append(np.eye(3))
            continue
        H, _ = optimize_homography(
            pts_A, means_B, covs_B, peaks_B=peaks_B,
            model="sRT", verbose=0, quiet=True,
            ransac_method=DEFAULT_RANSAC,
        )
        out.append(H)
    return out


def pipeline_torch_seq(items: List[torch.Tensor], device: str) -> List[np.ndarray]:
    """torch extraction + torch-LM refinement, per frame."""
    out = []
    for logits in items:
        logits_dev = logits.to(device, non_blocking=True)
        pts_A, means_B, peaks_B, covs_B = find_gaussians_torch(
            logits_dev, adaptive_gauss_fit=False, log_missing_gaussians=False,
            device=device,
        )
        if pts_A.shape[0] < 4:
            out.append(np.eye(3))
            continue
        H_init = _ransac_init_one(pts_A, peaks_B)
        # torch-LM expects torch inputs.
        pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).to(device)
        means_B_t = torch.from_numpy(means_B.astype(np.float64)).to(device)
        covs_B_t = torch.from_numpy(covs_B.astype(np.float64)).to(device)
        H_init_t = torch.from_numpy(H_init.astype(np.float64)).to(device)
        with torch.no_grad():
            H_opt = refine_homography_torch_lm_torch(
                pts_A_t, means_B_t, covs_B_t, H_init_t, model="sRT",
            )
        out.append(H_opt[0].detach().cpu().numpy())
    return out


def pipeline_torch_batched(
    stacked: torch.Tensor, device: str, mode: str
) -> List[np.ndarray]:
    """Batched extraction (one CUDA call) + RANSAC loop + LM (batched if same N)."""
    stacked_dev = stacked.to(device, non_blocking=True)
    per_frame = find_gaussians_torch_batch(stacked_dev, device=device)

    # RANSAC must run sequentially on CPU (cv2 has no batched API).
    H_inits = []
    for pts_A, _means_B, peaks_B, _covs_B in per_frame:
        if pts_A.shape[0] < 4:
            H_inits.append(np.eye(3))
        else:
            H_inits.append(_ransac_init_one(pts_A, peaks_B))

    # Refinement: batched if all frames share the same N, else loop.
    Ns = [t[0].shape[0] for t in per_frame]
    if mode == "homogeneous" and len(set(Ns)) == 1 and Ns[0] >= 4:
        N = Ns[0]
        pts_A = torch.from_numpy(
            np.stack([t[0] for t in per_frame]).astype(np.float64)
        ).to(device)
        means_B = torch.from_numpy(
            np.stack([t[1] for t in per_frame]).astype(np.float64)
        ).to(device)
        covs_B = torch.from_numpy(
            np.stack([t[3] for t in per_frame]).astype(np.float64)
        ).to(device)
        H_init_t = torch.from_numpy(np.stack(H_inits).astype(np.float64)).to(device)
        with torch.no_grad():
            H_opt = refine_homography_torch_lm_torch(
                pts_A, means_B, covs_B, H_init_t, model="sRT",
            )
        return [H_opt[i].detach().cpu().numpy() for i in range(len(per_frame))]

    # Heterogeneous (or any frame too small): loop the LM step.
    out = []
    for (pts_A, means_B, peaks_B, covs_B), H_init in zip(per_frame, H_inits):
        if pts_A.shape[0] < 4:
            out.append(np.eye(3))
            continue
        pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).to(device)
        means_B_t = torch.from_numpy(means_B.astype(np.float64)).to(device)
        covs_B_t = torch.from_numpy(covs_B.astype(np.float64)).to(device)
        H_init_t = torch.from_numpy(H_init.astype(np.float64)).to(device)
        with torch.no_grad():
            H_opt = refine_homography_torch_lm_torch(
                pts_A_t, means_B_t, covs_B_t, H_init_t, model="sRT",
            )
        out.append(H_opt[0].detach().cpu().numpy())
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        default=os.path.join(_REPO_ROOT, "tensors"),
        help="Path to .pt tensor dir; defaults to <repo>/tensors.",
    )
    ap.add_argument("--batch-sizes", default="1,4,16,64")
    ap.add_argument("--mode", choices=["homogeneous", "heterogeneous"], default="homogeneous")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-cuda", action="store_true",
                    help="Skip the CUDA paths even if CUDA is available.")
    ap.add_argument("--output", default=None,
                    help="JSON path; defaults to <repo>/benchmarks/results/e2e_batched_<mode>.json.")
    args = ap.parse_args()

    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    have_cuda = torch.cuda.is_available() and not args.no_cuda

    print(f"\nMode: {args.mode}, repeats={args.repeats}, warmup={args.warmup}, "
          f"cuda={'yes' if have_cuda else 'no'}")
    header = f"{'B':>4}  {'np_seq':>10}  {'torch_seq_cpu':>14}"
    if have_cuda:
        header += f"  {'torch_seq_cuda':>15}  {'torch_batched':>14}"
    header += f"  {'best_speedup':>13}"
    print(header)
    print("-" * len(header))

    rows: List[Dict[str, Any]] = []

    for B in sizes:
        items, stacked = _make_batch(args.dataset_dir, B, args.mode)

        t_np = _time(lambda: pipeline_np_seq(items),
                     args.repeats, args.warmup, "cpu")
        t_seq_cpu = _time(lambda: pipeline_torch_seq(items, "cpu"),
                          args.repeats, args.warmup, "cpu")

        if have_cuda:
            stacked_cu = stacked.cuda()
            items_cu = [t.cuda() for t in items]
            t_seq_cu = _time(lambda: pipeline_torch_seq(items_cu, "cuda"),
                             args.repeats, args.warmup, "cuda")
            t_batched = _time(lambda: pipeline_torch_batched(stacked_cu, "cuda", args.mode),
                              args.repeats, args.warmup, "cuda")
        else:
            t_seq_cu = t_batched = [float("nan")]

        med_np = float(statistics.median(t_np))
        med_seq_cpu = float(statistics.median(t_seq_cpu))
        med_seq_cu = float(statistics.median(t_seq_cu))
        med_batched = float(statistics.median(t_batched))

        # Best speedup vs numpy baseline (for the table at-a-glance).
        if have_cuda:
            best = med_np / med_batched if med_batched > 0 else float("inf")
        else:
            best = med_np / med_seq_cpu if med_seq_cpu > 0 else float("inf")

        line = f"{B:>4}  {med_np:>10.2f}  {med_seq_cpu:>14.2f}"
        if have_cuda:
            line += f"  {med_seq_cu:>15.2f}  {med_batched:>14.2f}"
        line += f"  {best:>12.2f}x"
        print(line)

        row = dict(
            batch_size=B,
            mode=args.mode,
            np_seq_ms_median=med_np,
            np_seq_ms_p95=float(np.percentile(t_np, 95)),
            torch_seq_cpu_ms_median=med_seq_cpu,
            torch_seq_cpu_ms_p95=float(np.percentile(t_seq_cpu, 95)),
        )
        if have_cuda:
            row.update(
                torch_seq_cuda_ms_median=med_seq_cu,
                torch_seq_cuda_ms_p95=float(np.percentile(t_seq_cu, 95)),
                torch_batched_ms_median=med_batched,
                torch_batched_ms_p95=float(np.percentile(t_batched, 95)),
                speedup_batched_vs_numpy=med_np / med_batched if med_batched > 0 else float("inf"),
                speedup_batched_vs_seq_cuda=med_seq_cu / med_batched if med_batched > 0 else float("inf"),
            )
        rows.append(row)

    out_path = args.output or os.path.join(
        _REPO_ROOT, "benchmarks", "results", f"e2e_batched_{args.mode}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "rows": rows}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
