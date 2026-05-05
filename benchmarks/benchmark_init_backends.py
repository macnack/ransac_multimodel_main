"""Compare init_backend choices: cv2 RANSAC vs kornia DLT vs torch RANSAC.

Times the H_init step alone (NOT the full pipeline — refinement is the
same regardless of backend). For each batch size we report the per-call
median wall time and the corner-error of the resulting init under the
ground-truth dataloader-space homography.

Run from any CWD::

    PYTHONPATH=<repo> python -m benchmarks.benchmark_init_backends \
        --sample-ids 98,122,128 --batch-sizes 1,16,64 --device cuda
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
from ransac_multimodel.dlt_ransac import (  # noqa: E402
    _KORNIA_OK,
    dlt_homography_kornia,
    torch_ransac_homography,
)
from ransac_multimodel.homography import compute_corner_error  # noqa: E402
from ransac_multimodel.transforms import (  # noqa: E402
    convert_to_dataloader_homography,
)


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


def _ce(H: np.ndarray, H_gt: np.ndarray) -> float:
    H_dl = convert_to_dataloader_homography(
        H / H[2, 2], 14, 64, crop_res=(224, 224), map_res=(896, 896),
    )
    return float(compute_corner_error(H_gt.astype(np.float64), H_dl, w=224, h=224))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        default=os.path.join(_REPO_ROOT, "tensors"),
    )
    ap.add_argument("--sample-ids", default="98,122,128")
    ap.add_argument("--batch-sizes", default="1,16,64")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--n-hypotheses", type=int, default=1024,
                    help="Number of RANSAC hypotheses for the torch_ransac path.")
    ap.add_argument("--output", default=os.path.join(
        _REPO_ROOT, "benchmarks", "results", "init_backends.json"))
    args = ap.parse_args()

    sids = [int(s) for s in args.sample_ids.split(",")]
    sizes = [int(s) for s in args.batch_sizes.split(",")]
    device = args.device

    # Pre-load + extract once per sample.
    samples = {}
    for sid in sids:
        t = torch.load(
            os.path.join(args.dataset_dir, f"sample_{sid:03d}_tensor.pt"),
            map_location="cpu",
        )
        gt = torch.load(
            os.path.join(args.dataset_dir, f"input_sample_{sid:06d}.pt"),
            map_location="cpu",
        )
        H_gt = gt["homography_gt"]
        H_gt = H_gt.numpy() if hasattr(H_gt, "numpy") else np.asarray(H_gt)
        logits = t[16]["gm_cls"][0]
        pts_A, means_B, peaks_B, _ = find_gaussians(
            logits, adaptive_gauss_fit=False, log_missing_gaussians=False,
        )
        samples[sid] = (pts_A, means_B, peaks_B, H_gt)
        print(f"sample {sid}: N={pts_A.shape[0]}")

    rows: List[Dict[str, Any]] = []
    print()
    print(f"{'B':>4}  {'sample':>6}  {'cv2 ms':>8}  {'cv2 err':>8}  "
          f"{'k_dlt ms':>10}  {'k_dlt err':>10}  "
          f"{'k_iter ms':>10}  {'k_iter err':>10}  "
          f"{'torch_rs ms':>11}  {'torch_rs err':>12}")
    print("-" * 110)

    for B in sizes:
        for sid in sids:
            pts_A, means_B, peaks_B, H_gt = samples[sid]
            # Replicate B times to test batched throughput.
            pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).unsqueeze(0).expand(B, -1, -1).to(device).contiguous()
            pts_B_t = torch.from_numpy(peaks_B.astype(np.float64)).unsqueeze(0).expand(B, -1, -1).to(device).contiguous()

            # cv2 (sequential CPU loop — cv2 has no batched API)
            def cv2_call():
                for _ in range(B):
                    cv2.findHomography(
                        pts_A, peaks_B, cv2.USAC_FAST,
                        ransacReprojThreshold=3.0, maxIters=5000,
                    )
            t_cv2 = _time(cv2_call, args.repeats, args.warmup, "cpu")
            H_cv2, _ = cv2.findHomography(pts_A, peaks_B, cv2.USAC_FAST, maxIters=5000)
            err_cv2 = _ce(H_cv2, H_gt) if H_cv2 is not None else float("nan")

            # kornia DLT
            err_kdlt = err_kiter = float("nan")
            t_kdlt = t_kiter = [float("nan")]
            if _KORNIA_OK:
                def k_dlt_call():
                    with torch.no_grad():
                        dlt_homography_kornia(pts_A_t, pts_B_t)
                t_kdlt = _time(k_dlt_call, args.repeats, args.warmup, device)
                with torch.no_grad():
                    H_kdlt = dlt_homography_kornia(pts_A_t, pts_B_t)[0].cpu().numpy()
                err_kdlt = _ce(H_kdlt, H_gt)

                def k_iter_call():
                    with torch.no_grad():
                        dlt_homography_kornia(pts_A_t, pts_B_t, iterated=True)
                t_kiter = _time(k_iter_call, args.repeats, args.warmup, device)
                with torch.no_grad():
                    H_kiter = dlt_homography_kornia(pts_A_t, pts_B_t, iterated=True)[0].cpu().numpy()
                err_kiter = _ce(H_kiter, H_gt)

            # torch RANSAC
            def t_rs_call():
                with torch.no_grad():
                    torch_ransac_homography(
                        pts_A_t, pts_B_t,
                        n_hypotheses=args.n_hypotheses, seed=42,
                    )
            t_trs = _time(t_rs_call, args.repeats, args.warmup, device)
            with torch.no_grad():
                H_trs, _ = torch_ransac_homography(
                    pts_A_t, pts_B_t,
                    n_hypotheses=args.n_hypotheses, seed=42,
                )
            H_trs = H_trs[0].cpu().numpy()
            err_trs = _ce(H_trs, H_gt)

            print(f"{B:>4}  {sid:>6}  "
                  f"{statistics.median(t_cv2):>8.2f}  {err_cv2:>8.1f}  "
                  f"{statistics.median(t_kdlt):>10.2f}  {err_kdlt:>10.1f}  "
                  f"{statistics.median(t_kiter):>10.2f}  {err_kiter:>10.1f}  "
                  f"{statistics.median(t_trs):>11.2f}  {err_trs:>12.1f}")

            rows.append({
                "batch_size": B, "sample_id": sid, "device": device,
                "n_correspondences": int(pts_A.shape[0]),
                "cv2_ms_median": float(statistics.median(t_cv2)),
                "kornia_dlt_ms_median": float(statistics.median(t_kdlt)),
                "kornia_dlt_iter_ms_median": float(statistics.median(t_kiter)),
                "torch_ransac_ms_median": float(statistics.median(t_trs)),
                "cv2_corner_err_px": err_cv2,
                "kornia_dlt_corner_err_px": err_kdlt,
                "kornia_dlt_iter_corner_err_px": err_kiter,
                "torch_ransac_corner_err_px": err_trs,
            })

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
