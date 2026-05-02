"""Compare find_gaussians (numpy/cv2) vs find_gaussians_torch on real samples.

For each (sample_id, path) combination we report the number of correspondences
detected and the median / p95 wall time across ``--repeats`` runs (after
``--warmup`` runs that are discarded).

Three paths are timed:
  * ``numpy``     -- the reference :func:`ransac_multimodel.correspondence.find_gaussians`.
  * ``torch_cpu`` -- :func:`ransac_multimodel.correspondence_torch.find_gaussians_torch`
                     on the CPU.
  * ``torch_cuda``-- the same torch version on CUDA, when available (and when
                     ``--device`` is not ``cpu``).  CUDA timings call
                     ``torch.cuda.synchronize()`` immediately before and after
                     the timed call so we measure actual kernel work, not the
                     async launch.

Run from any CWD::

    PYTHONPATH=<repo> python -m benchmarks.benchmark_find_gaussians \
        --sample-ids 98,122,128 --device auto \
        --output benchmarks/results/find_gaussians_perf.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from ransac_multimodel.correspondence import find_gaussians  # noqa: E402
from ransac_multimodel.correspondence_torch import find_gaussians_torch  # noqa: E402


# --------------------------------------------------------------------------- #
# Sample loading                                                              #
# --------------------------------------------------------------------------- #

def parse_sample_ids(spec: str) -> List[int]:
    """Parse a comma-separated list of ints, with optional ``a-b`` ranges."""
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def load_logits(sample_id: int, dataset_dir: str) -> torch.Tensor:
    """Load the per-patch logits tensor for ``sample_id`` from ``dataset_dir``."""
    path = os.path.join(dataset_dir, f"sample_{sample_id:03d}_tensor.pt")
    sample = torch.load(path, map_location="cpu", weights_only=False)
    logits = sample[16]["gm_cls"][0]
    return logits


# --------------------------------------------------------------------------- #
# Timing helpers                                                              #
# --------------------------------------------------------------------------- #

def _time_call(
    fn: Callable[[], Any],
    repeats: int,
    warmup: int,
    sync_cuda: bool = False,
) -> tuple[Any, List[float]]:
    """Call ``fn`` repeatedly, returning the last result and per-call ms timings."""

    def _sync() -> None:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    out = None
    for _ in range(warmup):
        out = fn()
        _sync()
    times: List[float] = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return out, times


# --------------------------------------------------------------------------- #
# Per-sample runner                                                           #
# --------------------------------------------------------------------------- #

def run_sample(
    sample_id: int,
    logits: torch.Tensor,
    *,
    repeats: int,
    warmup: int,
    run_cuda: bool,
) -> Dict[str, Any]:
    """Run all configured paths against ``logits`` and return a row dict."""

    # numpy / cv2 reference
    def numpy_call():
        return find_gaussians(
            logits,
            adaptive_gauss_fit=False,
            log_missing_gaussians=False,
        )

    (np_out, np_times) = _time_call(numpy_call, repeats, warmup, sync_cuda=False)
    pts_A_np, _means_np, _peaks_np, _covs_np = np_out
    n_corresp_np = int(pts_A_np.shape[0])

    # torch CPU
    logits_cpu = logits.to(device="cpu")

    def torch_cpu_call():
        return find_gaussians_torch(
            logits_cpu,
            adaptive_gauss_fit=False,
            log_missing_gaussians=False,
            device="cpu",
        )

    (tcpu_out, tcpu_times) = _time_call(
        torch_cpu_call, repeats, warmup, sync_cuda=False
    )
    pts_A_t, _means_t, _peaks_t, _covs_t = tcpu_out
    n_corresp_torch = int(pts_A_t.shape[0])

    # torch CUDA (optional)
    tcuda_times: Optional[List[float]] = None
    if run_cuda:
        logits_cuda = logits.cuda()

        def torch_cuda_call():
            return find_gaussians_torch(
                logits_cuda,
                adaptive_gauss_fit=False,
                log_missing_gaussians=False,
                device="cuda",
            )

        (_tcuda_out, tcuda_times) = _time_call(
            torch_cuda_call, repeats, warmup, sync_cuda=True
        )

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "n_corresp_np": n_corresp_np,
        "n_corresp_torch": n_corresp_torch,
        "numpy_ms_median": float(statistics.median(np_times)),
        "numpy_ms_p95": float(np.percentile(np_times, 95)),
        "torch_cpu_ms_median": float(statistics.median(tcpu_times)),
        "torch_cpu_ms_p95": float(np.percentile(tcpu_times, 95)),
    }
    if tcuda_times is not None:
        row["torch_cuda_ms_median"] = float(statistics.median(tcuda_times))
        row["torch_cuda_ms_p95"] = float(np.percentile(tcuda_times, 95))
    return row


# --------------------------------------------------------------------------- #
# Console table                                                                #
# --------------------------------------------------------------------------- #

def _print_table(rows: List[Dict[str, Any]], have_cuda: bool) -> None:
    if have_cuda:
        fmt = (
            "{:>8}  {:>6}  {:>9}  {:>9}  {:>11}  {:>11}  {:>13}  {:>14}"
        )
        print(
            fmt.format(
                "sample",
                "N_np",
                "N_torch",
                "np_ms",
                "t_cpu_ms",
                "t_cuda_ms",
                "cpu_speedup",
                "cuda_speedup",
            )
        )
        print("-" * 100)
        for r in rows:
            np_ms = r["numpy_ms_median"]
            cpu_ms = r["torch_cpu_ms_median"]
            cuda_ms = r["torch_cuda_ms_median"]
            cpu_sp = np_ms / cpu_ms if cpu_ms > 0 else float("inf")
            cuda_sp = np_ms / cuda_ms if cuda_ms > 0 else float("inf")
            print(
                fmt.format(
                    r["sample_id"],
                    r["n_corresp_np"],
                    r["n_corresp_torch"],
                    f"{np_ms:.2f}",
                    f"{cpu_ms:.2f}",
                    f"{cuda_ms:.2f}",
                    f"{cpu_sp:.2f}x",
                    f"{cuda_sp:.2f}x",
                )
            )
    else:
        fmt = "{:>8}  {:>6}  {:>9}  {:>9}  {:>11}  {:>13}"
        print(
            fmt.format(
                "sample",
                "N_np",
                "N_torch",
                "np_ms",
                "t_cpu_ms",
                "cpu_speedup",
            )
        )
        print("-" * 70)
        for r in rows:
            np_ms = r["numpy_ms_median"]
            cpu_ms = r["torch_cpu_ms_median"]
            cpu_sp = np_ms / cpu_ms if cpu_ms > 0 else float("inf")
            print(
                fmt.format(
                    r["sample_id"],
                    r["n_corresp_np"],
                    r["n_corresp_torch"],
                    f"{np_ms:.2f}",
                    f"{cpu_ms:.2f}",
                    f"{cpu_sp:.2f}x",
                )
            )


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="./tensors")
    ap.add_argument(
        "--sample-ids",
        default="98,122,128",
        help="comma list, range like 98-100",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="auto = run torch CUDA when available, cpu = skip CUDA path",
    )
    ap.add_argument(
        "--output",
        default="benchmarks/results/find_gaussians_perf.json",
    )
    args = ap.parse_args()

    have_cuda = args.device == "auto" and torch.cuda.is_available()

    sample_ids = parse_sample_ids(args.sample_ids)

    rows: List[Dict[str, Any]] = []
    samples_payload: Dict[str, Dict[str, Any]] = {}
    for sid in sample_ids:
        try:
            logits = load_logits(sid, args.dataset_dir)
        except FileNotFoundError as e:
            print(f"[skip] sample {sid}: {e}")
            continue

        row = run_sample(
            sid,
            logits,
            repeats=args.repeats,
            warmup=args.warmup,
            run_cuda=have_cuda,
        )
        rows.append(row)
        # JSON keys per the spec are stringified sample ids.
        sample_entry = {k: v for k, v in row.items() if k != "sample_id"}
        samples_payload[str(sid)] = sample_entry

    _print_table(rows, have_cuda=have_cuda)

    payload = {
        "repeats": args.repeats,
        "warmup": args.warmup,
        "device": "cuda" if have_cuda else "cpu",
        "samples": samples_payload,
    }
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
