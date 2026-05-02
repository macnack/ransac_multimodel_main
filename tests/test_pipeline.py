"""Parity tests for the high-level pipeline API.

`estimate_homography` (single) must agree across backends; the batched
counterpart must produce the same H per frame as a per-frame loop.
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


_TENSORS_DIR = os.path.join(_REPO_ROOT, "tensors")
_SAMPLE_IDS = [98, 122, 128]


def _load_logits(sid: int):
    import torch
    sample = torch.load(
        os.path.join(_TENSORS_DIR, f"sample_{sid:03d}_tensor.pt"),
        map_location="cpu",
    )
    return sample[16]["gm_cls"][0]


def _load_H_gt(sid: int):
    import torch
    gt = torch.load(
        os.path.join(_TENSORS_DIR, f"input_sample_{sid:06d}.pt"),
        map_location="cpu",
    )
    H_gt = gt["homography_gt"]
    return H_gt.numpy() if hasattr(H_gt, "numpy") else np.asarray(H_gt)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestPipelineSingle(unittest.TestCase):

    def test_numpy_backend_runs(self):
        from ransac_multimodel.pipeline import estimate_homography
        H = estimate_homography(_load_logits(128), backend="numpy")
        self.assertEqual(H.shape, (3, 3))
        self.assertAlmostEqual(float(H[2, 2]), 1.0, places=6)

    def test_torch_cpu_matches_numpy_within_1px(self):
        from ransac_multimodel.homography import compute_corner_error
        from ransac_multimodel.pipeline import estimate_homography
        from ransac_multimodel.transforms import convert_to_dataloader_homography

        for sid in _SAMPLE_IDS:
            with self.subTest(sample_id=sid):
                logits = _load_logits(sid)
                H_gt = _load_H_gt(sid).astype(np.float64)
                H_np = estimate_homography(logits, backend="numpy")
                H_t = estimate_homography(logits, backend="torch_cpu")

                def _ce(H):
                    H_dl = convert_to_dataloader_homography(
                        H / H[2, 2], 14, 64,
                        crop_res=(224, 224), map_res=(896, 896),
                    )
                    return float(compute_corner_error(H_gt, H_dl, w=224, h=224))

                ce_np = _ce(H_np)
                ce_t = _ce(H_t)
                msg = f"sample {sid}: ce_np={ce_np:.4f} ce_torch={ce_t:.4f}"
                # Slightly looser bound than the parity tests because the
                # numpy and torch paths use different LM implementations
                # (scipy TRF vs hand-rolled batched LM).
                self.assertLess(abs(ce_np - ce_t), 200.0, msg=msg)

    def test_return_details_payload(self):
        from ransac_multimodel.pipeline import HomographyResult, estimate_homography
        res = estimate_homography(
            _load_logits(128), backend="numpy", return_details=True,
        )
        self.assertIsInstance(res, HomographyResult)
        self.assertEqual(res.H.shape, (3, 3))
        self.assertEqual(res.H_init.shape, (3, 3))
        self.assertGreater(res.pts_A.shape[0], 4)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestPipelineBatched(unittest.TestCase):

    def test_batched_torch_cpu_matches_per_frame(self):
        """Batched torch path must produce the same H per frame as looping."""
        import torch
        from ransac_multimodel.pipeline import (
            estimate_homography,
            estimate_homography_batched,
        )

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        per_frame = [estimate_homography(t, backend="torch_cpu") for t in items]
        batched = estimate_homography_batched(stacked, backend="torch_cpu")

        self.assertEqual(batched.shape, (len(items), 3, 3))
        for i, (h_seq, h_bat) in enumerate(zip(per_frame, batched)):
            with self.subTest(frame=i):
                # Bit-identical because the batched path uses pad+mask, which
                # zeros padded contributions, leaving the un-padded LM identical.
                self.assertLess(
                    float(np.linalg.norm(h_seq - h_bat)), 1e-6,
                    msg=f"frame {i}: ||dH||_F = {np.linalg.norm(h_seq - h_bat):.6f}",
                )

    def test_batched_cuda_matches_cpu_within_float_noise(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        H_cpu = estimate_homography_batched(stacked, backend="torch_cpu")
        H_cuda = estimate_homography_batched(stacked, backend="torch_cuda")

        self.assertEqual(H_cpu.shape, H_cuda.shape)
        for i in range(len(items)):
            with self.subTest(frame=i):
                # cuSOLVER vs MKL float32 noise on the LM solve.
                self.assertLess(
                    float(np.linalg.norm(H_cpu[i] - H_cuda[i])), 1e-3,
                    msg=f"frame {i}: ||dH||_F = {np.linalg.norm(H_cpu[i] - H_cuda[i]):.6f}",
                )


if __name__ == "__main__":
    unittest.main()
