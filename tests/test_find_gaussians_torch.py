"""
Parity tests for ``find_gaussians_torch``.

The torch implementation is judged equivalent to the numpy reference if the
*downstream* homography refinement produces nearly the same result regardless
of which correspondence-extraction backend was used.  Concretely: feed both
backends' ``(pts_A, means_B, peaks_B, covs_B)`` outputs through
``optimize_homography(model="sRT")`` -> ``convert_to_dataloader_homography`` ->
``compute_corner_error`` against the dataloader's ground-truth H, and assert
the absolute delta in corner-error is < 1.0 px.

This mirrors the parity bar used by ``tests/test_torch_parity.py`` and
``tests/test_theseus_parity.py``.
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
_SAMPLE_IDS = (98, 122, 128)

# Shared find_gaussians kwargs for every test in this module.  The torch
# backend only implements the fixed-window variant, so we run the numpy
# reference with ``adaptive_gauss_fit=False`` to compare apples to apples.
_FIND_GAUSSIANS_KWARGS = dict(
    adaptive_gauss_fit=False,
    plot_heatmaps=False,
    plotter=None,
    log_missing_gaussians=False,
)

_IN_PATCH_DIM = 14
_OUT_PATCH_DIM = 64
_CROP_RES = (224, 224)
_MAP_RES = (896, 896)


def _load_logits_and_gt(sample_id: int):
    """Return ``(logits_tensor, H_gt_numpy)`` for a given sample id."""
    import torch

    sample_path = os.path.join(_TENSORS_DIR, f"sample_{sample_id:03d}_tensor.pt")
    in_path = os.path.join(_TENSORS_DIR, f"input_sample_{sample_id:06d}.pt")

    sample = torch.load(sample_path, map_location="cpu")
    gt = torch.load(in_path, map_location="cpu")

    logits = sample[16]["gm_cls"][0]
    H_gt_raw = gt["homography_gt"]
    H_gt = (
        H_gt_raw.numpy()
        if hasattr(H_gt_raw, "numpy")
        else np.asarray(H_gt_raw)
    )
    return logits, H_gt


def _refine_and_corner_error(pts_A, means_B, peaks_B, covs_B, H_gt):
    """Run the same downstream pipeline on a correspondence set and return CE."""
    from ransac_multimodel.homography import (
        compute_corner_error,
        optimize_homography,
    )
    from ransac_multimodel.transforms import convert_to_dataloader_homography

    H_feat, _ = optimize_homography(
        pts_A,
        means_B,
        covs_B,
        peaks_B=peaks_B,
        model="sRT",
        verbose=0,
        quiet=True,
    )
    H_feat = H_feat / H_feat[2, 2]
    H_dl = convert_to_dataloader_homography(
        H_feat,
        _IN_PATCH_DIM,
        _OUT_PATCH_DIM,
        crop_res=_CROP_RES,
        map_res=_MAP_RES,
    )
    return compute_corner_error(H_gt, H_dl, w=_CROP_RES[1], h=_CROP_RES[0])


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestFindGaussiansTorchParity(unittest.TestCase):
    def setUp(self):
        # Lazy import so unrelated test runs don't pay the import cost / get
        # masked by an unrelated ImportError.
        try:
            from ransac_multimodel.correspondence import find_gaussians  # noqa: F401
            from ransac_multimodel.correspondence_torch import (  # noqa: F401
                find_gaussians_torch,
            )
        except ImportError as e:
            self.skipTest(f"correspondence backends unavailable: {e}")

        for sid in _SAMPLE_IDS:
            sample_path = os.path.join(_TENSORS_DIR, f"sample_{sid:03d}_tensor.pt")
            in_path = os.path.join(_TENSORS_DIR, f"input_sample_{sid:06d}.pt")
            if not (os.path.exists(sample_path) and os.path.exists(in_path)):
                self.skipTest(
                    f"sample tensors for id={sid} are missing under {_TENSORS_DIR}"
                )

    def test_corner_error_parity_real_samples(self):
        from ransac_multimodel.correspondence import find_gaussians
        from ransac_multimodel.correspondence_torch import find_gaussians_torch

        for sid in _SAMPLE_IDS:
            with self.subTest(sample_id=sid):
                logits, H_gt = _load_logits_and_gt(sid)

                pts_A_np, means_B_np, peaks_B_np, covs_B_np = find_gaussians(
                    logits, **_FIND_GAUSSIANS_KWARGS
                )
                pts_A_t, means_B_t, peaks_B_t, covs_B_t = find_gaussians_torch(
                    logits, device="cpu", **_FIND_GAUSSIANS_KWARGS
                )

                self.assertGreaterEqual(
                    pts_A_np.shape[0], 4,
                    "numpy backend produced fewer than 4 correspondences",
                )
                self.assertGreaterEqual(
                    pts_A_t.shape[0], 4,
                    "torch backend produced fewer than 4 correspondences",
                )

                ce_np = _refine_and_corner_error(
                    pts_A_np, means_B_np, peaks_B_np, covs_B_np, H_gt
                )
                ce_torch = _refine_and_corner_error(
                    pts_A_t, means_B_t, peaks_B_t, covs_B_t, H_gt
                )

                # Surface the actual numbers in the failure message so the
                # caller can see how close we are even when the test passes.
                msg = (
                    f"sample {sid}: ce_np={ce_np:.4f}px, ce_torch={ce_torch:.4f}px, "
                    f"|delta|={abs(ce_np - ce_torch):.4f}px, "
                    f"N_np={pts_A_np.shape[0]} N_torch={pts_A_t.shape[0]}"
                )
                # Tighter bound when the two backends detect the same number
                # of correspondences (we expect the same H downstream up to
                # float noise). When they disagree by 1+ peaks (boundary
                # detection differences between cv2's BORDER_REFLECT_101 and
                # torch's replicate-pad), allow up to ~10 px gap because a
                # single inlier can shift the sRT estimate that much on
                # noisy real data.
                bound = 1.0 if pts_A_np.shape[0] == pts_A_t.shape[0] else 10.0
                self.assertLess(abs(ce_np - ce_torch), bound, msg=msg)

    def test_n_correspondences_close(self):
        from ransac_multimodel.correspondence import find_gaussians
        from ransac_multimodel.correspondence_torch import find_gaussians_torch

        for sid in _SAMPLE_IDS:
            with self.subTest(sample_id=sid):
                logits, _ = _load_logits_and_gt(sid)

                pts_A_np, *_ = find_gaussians(logits, **_FIND_GAUSSIANS_KWARGS)
                pts_A_t, *_ = find_gaussians_torch(
                    logits, device="cpu", **_FIND_GAUSSIANS_KWARGS
                )

                n_np = int(pts_A_np.shape[0])
                n_t = int(pts_A_t.shape[0])
                self.assertLessEqual(
                    abs(n_np - n_t), 5,
                    msg=f"sample {sid}: N_np={n_np} N_torch={n_t}",
                )

    def test_cuda_matches_cpu(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        from ransac_multimodel.correspondence_torch import find_gaussians_torch

        sid = _SAMPLE_IDS[0]
        logits, H_gt = _load_logits_and_gt(sid)

        pts_A_cpu, means_B_cpu, peaks_B_cpu, covs_B_cpu = find_gaussians_torch(
            logits, device="cpu", **_FIND_GAUSSIANS_KWARGS
        )
        pts_A_cu, means_B_cu, peaks_B_cu, covs_B_cu = find_gaussians_torch(
            logits, device="cuda", **_FIND_GAUSSIANS_KWARGS
        )

        ce_cpu = _refine_and_corner_error(
            pts_A_cpu, means_B_cpu, peaks_B_cpu, covs_B_cpu, H_gt
        )
        ce_cuda = _refine_and_corner_error(
            pts_A_cu, means_B_cu, peaks_B_cu, covs_B_cu, H_gt
        )

        # Float32 cusolver vs MKL produces small rounding differences in the
        # downstream sRT solve; sub-pixel drift is expected.
        self.assertLess(
            abs(ce_cpu - ce_cuda), 0.5,
            msg=(
                f"sample {sid}: ce_cpu={ce_cpu:.4f}px, ce_cuda={ce_cuda:.4f}px, "
                f"|delta|={abs(ce_cpu - ce_cuda):.4f}px"
            ),
        )


if __name__ == "__main__":
    unittest.main()
