"""Tests for the DLT/RANSAC alternatives to cv2.findHomography."""

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


_TENSORS_DIR = os.path.join(_REPO_ROOT, "tensors")
_SAMPLE_IDS = [98, 122, 128]


def _load_logits(sid: int):
    import torch
    sample = torch.load(
        os.path.join(_TENSORS_DIR, f"sample_{sid:03d}_tensor.pt"),
        map_location="cpu",
    )
    return sample[16]["gm_cls"][0]


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestDLTRansacLowLevel(unittest.TestCase):
    def test_torch_ransac_runs_without_kornia(self):
        """torch_ransac path must work even without kornia (no refine step)."""
        import torch
        from ransac_multimodel.dlt_ransac import torch_ransac_homography

        rng = np.random.default_rng(0)
        pts_A = rng.uniform(0, 14, size=(1, 50, 2)).astype(np.float64)
        pts_B = pts_A * 4 + rng.normal(0, 0.1, size=(1, 50, 2))
        H, mask = torch_ransac_homography(
            torch.from_numpy(pts_A), torch.from_numpy(pts_B),
            n_hypotheses=256, seed=42, refine_with_inliers=False,
        )
        self.assertEqual(H.shape, (1, 3, 3))
        self.assertEqual(mask.shape, (1, 50))
        # Synthetic noise-free correspondences -> almost all should be inliers.
        self.assertGreater(int(mask.sum()), 40)

    @unittest.skipUnless(_has_module("kornia"), "kornia not installed")
    def test_kornia_dlt_runs(self):
        import torch
        from ransac_multimodel.dlt_ransac import dlt_homography_kornia

        rng = np.random.default_rng(0)
        pts_A = rng.uniform(0, 14, size=(2, 50, 2)).astype(np.float64)
        pts_B = pts_A * 4 + rng.normal(0, 0.05, size=(2, 50, 2))
        H = dlt_homography_kornia(torch.from_numpy(pts_A), torch.from_numpy(pts_B))
        self.assertEqual(H.shape, (2, 3, 3))
        # H should be approximately a uniform 4x scaling for our synthetic case.
        # H[0,0] and H[1,1] should be ~4.
        self.assertAlmostEqual(float(H[0, 0, 0]), 4.0, delta=0.1)
        self.assertAlmostEqual(float(H[0, 1, 1]), 4.0, delta=0.1)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestPipelineInitBackends(unittest.TestCase):
    def test_all_init_backends_run_end_to_end(self):
        """Each init_backend must accept B real samples and produce a (B,3,3) H."""
        import torch
        from ransac_multimodel.pipeline import (
            INIT_BACKENDS, _KORNIA_OK, estimate_homography_batched,
        )

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        for backend_name in INIT_BACKENDS:
            with self.subTest(init_backend=backend_name):
                if backend_name.startswith("kornia") and not _KORNIA_OK:
                    self.skipTest("kornia not installed")
                H = estimate_homography_batched(
                    stacked, backend="torch_cpu", init_backend=backend_name,
                )
                self.assertEqual(H.shape, (len(items), 3, 3))
                # Sanity: H[2,2] normalized to ~1.
                for i in range(len(items)):
                    self.assertAlmostEqual(
                        float(H[i, 2, 2]), 1.0, places=4,
                        msg=f"{backend_name} frame {i}: H[2,2] != 1",
                    )

    def test_resolve_init_backend(self):
        from ransac_multimodel import (
            INIT_BACKENDS, resolve_init_backend,
        )
        for name in INIT_BACKENDS:
            try:
                self.assertEqual(resolve_init_backend(name), name)
            except ModuleNotFoundError:
                # kornia paths may legitimately raise when kornia is absent.
                self.assertTrue(name.startswith("kornia"))
        with self.assertRaises(ValueError):
            resolve_init_backend("not_a_real_backend")


if __name__ == "__main__":
    unittest.main()
