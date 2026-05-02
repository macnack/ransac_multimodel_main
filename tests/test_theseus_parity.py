"""
Parity & smoke tests for `optimize_homography_theseus`.

Mirrors the bar set by tests/test_torch_parity.py: the theseus refinement
must land within < 1.0 px corner-error of the scipy refinement on the
synthetic case used by benchmarks/build_synthetic_case.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestTheseusParity(unittest.TestCase):
    def setUp(self):
        # Import lazily so unrelated test runs don't pay the theseus startup cost.
        try:
            from ransac_multimodel.homography_theseus import optimize_homography_theseus  # noqa: F401
        except ImportError as e:
            self.skipTest(f"theseus stack unavailable: {e}")

    def test_srt_corner_error_close_to_scipy(self):
        from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case
        from ransac_multimodel.homography import (
            compute_corner_error,
            optimize_homography,
        )
        from ransac_multimodel.homography_theseus import optimize_homography_theseus

        case = build_synthetic_case(seed=1234)

        H_np, _ = optimize_homography(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model="sRT",
            verbose=0,
            quiet=True,
        )
        H_th, _ = optimize_homography_theseus(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model="sRT",
            max_iter=100,
            step_size=1.0,
            quiet=True,
        )

        H_np = H_np / H_np[2, 2]
        H_th = H_th / H_th[2, 2]

        ce_np = compute_corner_error(case.H_gt, H_np, w=1024, h=1024)
        ce_th = compute_corner_error(case.H_gt, H_th, w=1024, h=1024)

        # Bound matches the torch parity test bound.
        self.assertLess(abs(ce_np - ce_th), 1.0)
        # Sanity: theseus shouldn't blow up the absolute error either.
        self.assertLess(ce_th, 5.0)

    def test_full_model_runs(self):
        from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case
        from ransac_multimodel.homography_theseus import optimize_homography_theseus

        case = build_synthetic_case(seed=1234)
        H_th, H_init = optimize_homography_theseus(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model="full",
            max_iter=50,
            step_size=1.0,
            quiet=True,
        )
        self.assertEqual(H_th.shape, (3, 3))
        self.assertAlmostEqual(float(H_th[2, 2]), 1.0, places=6)
        self.assertEqual(H_init.shape, (3, 3))

    def test_backprop_through_layer(self):
        """The differentiable wrapper must produce gradients on its inputs."""
        import torch

        from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case
        from ransac_multimodel.homography_theseus import (
            refine_homography_theseus_torch,
        )

        case = build_synthetic_case(seed=1234)

        pts_A = torch.as_tensor(case.pts_A, dtype=torch.float64)
        means_B = torch.as_tensor(case.means_B, dtype=torch.float64).requires_grad_(True)
        covs_B = torch.as_tensor(case.covs_B, dtype=torch.float64)
        H_init = torch.as_tensor(np.eye(3), dtype=torch.float64)
        H_gt = torch.as_tensor(case.H_gt, dtype=torch.float64)

        H_opt = refine_homography_theseus_torch(
            pts_A,
            means_B,
            covs_B,
            H_init,
            model="sRT",
            max_iter=30,
            step_size=1.0,
        )
        # Squeeze batch dim for the loss.
        loss = ((H_opt[0] - H_gt) ** 2).sum()
        loss.backward()

        self.assertIsNotNone(means_B.grad)
        self.assertTrue(torch.isfinite(means_B.grad).all())
        # Some component of the gradient must be non-zero.
        self.assertGreater(float(means_B.grad.abs().sum()), 0.0)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestTorchLMParity(unittest.TestCase):
    def test_srt_corner_error_close_to_scipy(self):
        from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case
        from ransac_multimodel.homography import (
            compute_corner_error,
            optimize_homography,
        )
        from ransac_multimodel.homography_torch_lm import optimize_homography_torch_lm

        case = build_synthetic_case(seed=1234)
        H_np, _ = optimize_homography(
            case.pts_A, case.means_B, case.covs_B,
            peaks_B=case.peaks_B, model="sRT", verbose=0, quiet=True,
        )
        H_lm, _ = optimize_homography_torch_lm(
            case.pts_A, case.means_B, case.covs_B,
            peaks_B=case.peaks_B, model="sRT",
        )
        ce_np = compute_corner_error(case.H_gt, H_np, w=1024, h=1024)
        ce_lm = compute_corner_error(case.H_gt, H_lm, w=1024, h=1024)
        self.assertLess(abs(ce_np - ce_lm), 1.0)
        self.assertLess(ce_lm, 5.0)

    def test_batched_runs(self):
        import torch

        from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case
        from ransac_multimodel.homography_torch_lm import refine_homography_torch_lm_torch

        case = build_synthetic_case(seed=1234)
        B = 4
        pts_A = torch.as_tensor(case.pts_A, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).contiguous()
        means_B = torch.as_tensor(case.means_B, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).contiguous()
        covs_B = torch.as_tensor(case.covs_B, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        H_init = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).contiguous()

        H_opt = refine_homography_torch_lm_torch(pts_A, means_B, covs_B, H_init, model="sRT")
        self.assertEqual(H_opt.shape, (B, 3, 3))

    def test_padded_masked_matches_per_frame(self):
        """pad_for_batched_lm + masked LM must match per-frame results."""
        import cv2
        import torch
        import numpy as np

        from ransac_multimodel.correspondence import find_gaussians
        from ransac_multimodel.homography_torch_lm import (
            pad_for_batched_lm,
            refine_homography_torch_lm_torch,
        )

        sids = [98, 122, 128]
        per_frame = []
        H_inits = []
        for sid in sids:
            sample = torch.load(
                os.path.join(_REPO_ROOT, "tensors", f"sample_{sid:03d}_tensor.pt"),
                map_location="cpu",
            )
            logits = sample[16]["gm_cls"][0]
            out = find_gaussians(
                logits, adaptive_gauss_fit=False, log_missing_gaussians=False,
            )
            per_frame.append(out)
            pts_A, _, peaks_B, _ = out
            H, _ = cv2.findHomography(pts_A, peaks_B, cv2.USAC_FAST, maxIters=5000)
            H_inits.append(H / H[2, 2])

        # Per-frame baseline
        per_frame_results = []
        for (pts_A, means_B, _peaks_B, covs_B), H_init in zip(per_frame, H_inits):
            pts = torch.from_numpy(pts_A.astype(np.float64))
            mu = torch.from_numpy(means_B.astype(np.float64))
            cov = torch.from_numpy(covs_B.astype(np.float64))
            Hi = torch.from_numpy(H_init)
            H = refine_homography_torch_lm_torch(pts, mu, cov, Hi, model="sRT")
            per_frame_results.append(H[0].numpy())

        # Padded batched
        pts_A_b, means_B_b, covs_B_b, H_init_b, mask = pad_for_batched_lm(
            per_frame, H_inits, device="cpu", dtype=torch.float64,
        )
        H_batched = refine_homography_torch_lm_torch(
            pts_A_b, means_B_b, covs_B_b, H_init_b, mask=mask, model="sRT",
        )

        for i, (h_seq, h_bat) in enumerate(zip(per_frame_results, H_batched.numpy())):
            with self.subTest(frame=i):
                # Bit-identical because masked LM does the same arithmetic
                # restricted to the unpadded slots.
                self.assertLess(
                    float(np.linalg.norm(h_seq - h_bat)), 1e-6,
                    msg=f"frame {i}: ||H_seq - H_bat||_F = {np.linalg.norm(h_seq - h_bat):.6f}",
                )


if __name__ == "__main__":
    unittest.main()
