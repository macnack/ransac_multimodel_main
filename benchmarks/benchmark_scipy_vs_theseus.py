"""
Compare scipy.optimize.least_squares vs theseus LM on identical inputs.

For each case we report:
  - corner error vs ground-truth homography (mean L2 over 4 corners, in
    feature-grid pixels of image B's patch grid),
  - end-to-end wall time in milliseconds (excluding the shared
    cv2.findHomography RANSAC initialization),
  - ||H_scipy - H_theseus||_F as a sanity check on agreement.

Run from any CWD:
    PYTHONPATH=<repo> python -m benchmarks.benchmark_scipy_vs_theseus \
        --modes residual --device cpu --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.benchmark_numpy_vs_torch import (  # noqa: E402
    SampleCase,
    build_synthetic_case,
    load_case_from_tensors,
    parse_sample_ids,
)
from ransac_multimodel.homography import (  # noqa: E402
    compute_corner_error,
    optimize_homography,
)
from ransac_multimodel.homography_theseus import (  # noqa: E402
    optimize_homography_theseus,
)
from ransac_multimodel.transforms import (  # noqa: E402
    convert_to_dataloader_homography,
)


@dataclass
class TimedResult:
    H: np.ndarray
    H_init: np.ndarray
    times_ms: List[float]
    corner_err: Optional[float]


def _time_call(fn, repeats: int, warmup: int, sync_cuda: bool = False):
    import torch as _torch  # local to keep top-level numpy-only imports clean

    def _sync():
        if sync_cuda and _torch.cuda.is_available():
            _torch.cuda.synchronize()

    for _ in range(warmup):
        out = fn()
        _sync()
    times = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return out, times


def run_case(
    case: SampleCase,
    *,
    model: str,
    repeats: int,
    warmup: int,
    f_scale: float,
    max_iter_scipy: int,
    max_iter_theseus: int,
    step_size_theseus: float,
    img_w: int,
    img_h: int,
    theseus_device: str = "cpu",
) -> Dict[str, Any]:
    # scipy
    def scipy_call():
        return optimize_homography(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model=model,
            verbose=0,
            quiet=True,
            f_scale=f_scale,
            max_nfev=max_iter_scipy,
        )

    (H_scipy, H_init_scipy), t_scipy = _time_call(scipy_call, repeats, warmup, sync_cuda=False)

    # theseus
    def theseus_call():
        return optimize_homography_theseus(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model=model,
            quiet=True,
            f_scale=f_scale,
            max_iter=max_iter_theseus,
            step_size=step_size_theseus,
            device=theseus_device,
        )

    (H_theseus, H_init_theseus), t_theseus = _time_call(
        theseus_call, repeats, warmup, sync_cuda=theseus_device.startswith("cuda")
    )

    H_scipy_n = H_scipy / H_scipy[2, 2]
    H_theseus_n = H_theseus / H_theseus[2, 2]
    H_init_n = H_init_scipy / H_init_scipy[2, 2]

    ce_scipy = ce_theseus = ce_init = None
    if case.H_gt is not None:
        H_gt = case.H_gt.astype(np.float64)
        if case.sample_id >= 0:
            # Real samples: feature-grid H -> dataloader-space H to compare
            # against the loader's GT, mirroring solve.py's path.
            in_patch_dim = 14
            out_patch_dim = 64
            crop_res = (224, 224)
            map_res = (896, 896)

            def _ce(H_feat):
                try:
                    H_dl = convert_to_dataloader_homography(
                        H_feat,
                        in_patch_dim,
                        out_patch_dim,
                        crop_res=crop_res,
                        map_res=map_res,
                    )
                except np.linalg.LinAlgError:
                    return float("nan")
                return float(compute_corner_error(H_gt, H_dl, w=crop_res[0], h=crop_res[1]))

            ce_scipy = _ce(H_scipy_n)
            ce_theseus = _ce(H_theseus_n)
            ce_init = _ce(H_init_n)
        else:
            # Synthetic case: H_gt already lives in feature-grid space.
            ce_scipy = float(compute_corner_error(H_gt, H_scipy_n, w=img_w, h=img_h))
            ce_theseus = float(compute_corner_error(H_gt, H_theseus_n, w=img_w, h=img_h))
            ce_init = float(compute_corner_error(H_gt, H_init_n, w=img_w, h=img_h))

    return {
        "sample_id": case.sample_id,
        "n_points": int(case.pts_A.shape[0]),
        "model": model,
        "scipy": {
            "time_ms_median": float(statistics.median(t_scipy)),
            "time_ms_p95": float(np.percentile(t_scipy, 95)),
            "corner_error_px": ce_scipy,
        },
        "theseus": {
            "time_ms_median": float(statistics.median(t_theseus)),
            "time_ms_p95": float(np.percentile(t_theseus, 95)),
            "corner_error_px": ce_theseus,
        },
        "ransac_init": {"corner_error_px": ce_init},
        "H_diff_frobenius": float(np.linalg.norm(H_scipy_n - H_theseus_n)),
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    fmt = "{:>10}  {:>5}  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}"
    print(
        fmt.format(
            "sample",
            "model",
            "N",
            "scipy ms",
            "thes ms",
            "scipy px",
            "thes px",
            "init px",
            "||dH||F",
        )
    )
    print("-" * 100)
    for r in rows:
        print(
            fmt.format(
                r["sample_id"],
                r["model"],
                r["n_points"],
                f"{r['scipy']['time_ms_median']:.2f}",
                f"{r['theseus']['time_ms_median']:.2f}",
                "n/a" if r["scipy"]["corner_error_px"] is None else f"{r['scipy']['corner_error_px']:.3f}",
                "n/a" if r["theseus"]["corner_error_px"] is None else f"{r['theseus']['corner_error_px']:.3f}",
                "n/a" if r["ransac_init"]["corner_error_px"] is None else f"{r['ransac_init']['corner_error_px']:.3f}",
                f"{r['H_diff_frobenius']:.4g}",
            )
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="./tensors")
    ap.add_argument(
        "--sample-ids",
        default="98,122,128",
        help="comma list, range like 98-100, or 'synthetic' for the built-in synthetic case only",
    )
    ap.add_argument("--include-synthetic", action="store_true", help="also run the synthetic case")
    ap.add_argument("--model", choices=["sRT", "full"], default="sRT")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--f-scale", type=float, default=2.0)
    ap.add_argument("--max-iter-scipy", type=int, default=5000)
    ap.add_argument("--max-iter-theseus", type=int, default=100)
    ap.add_argument("--step-size-theseus", type=float, default=1.0)
    ap.add_argument("--img-w", type=int, default=64, help="reference image B grid width for corner error")
    ap.add_argument("--img-h", type=int, default=64, help="reference image B grid height for corner error")
    ap.add_argument("--output", default="benchmarks/results/scipy_vs_theseus_latest.json")
    ap.add_argument("--theseus-device", default="cpu", help="cpu or cuda")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cases: List[SampleCase] = []

    if args.sample_ids == "synthetic" or args.include_synthetic:
        cases.append(build_synthetic_case(seed=1234))

    if args.sample_ids != "synthetic":
        ids = parse_sample_ids(args.sample_ids)
        for sid in ids:
            try:
                cases.append(load_case_from_tensors(sid, args.dataset_dir, quiet=args.quiet))
            except FileNotFoundError as e:
                print(f"[skip] sample {sid}: {e}")

    rows = []
    for case in cases:
        r = run_case(
            case,
            model=args.model,
            repeats=args.repeats,
            warmup=args.warmup,
            f_scale=args.f_scale,
            max_iter_scipy=args.max_iter_scipy,
            max_iter_theseus=args.max_iter_theseus,
            step_size_theseus=args.step_size_theseus,
            img_w=args.img_w,
            img_h=args.img_h,
            theseus_device=args.theseus_device,
        )
        rows.append(r)

    _print_table(rows)

    payload = {
        "model": args.model,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "f_scale": args.f_scale,
        "max_iter_scipy": args.max_iter_scipy,
        "max_iter_theseus": args.max_iter_theseus,
        "step_size_theseus": args.step_size_theseus,
        "rows": rows,
    }
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
