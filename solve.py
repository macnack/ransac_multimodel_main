import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from ransac_multimodel.correspondence import find_gaussians
from ransac_multimodel.homography import compute_corner_error, optimize_homography
from ransac_multimodel.parity_utils import gaussian_config, optimize_params
from ransac_multimodel.plotting import (
    plot_correspondences_with_arrows,
    plot_heatmap_comparison,
    plot_homography_projection,
    plot_image_homography_warp,
)
from ransac_multimodel.transforms import (
    convert_to_dataloader_homography,
    convert_to_pixel_homography,
)


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./solve_config.json", help="Path to JSON config")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.cfg))

    # settings
    mode = "sRT"  # 'full' or 'sRT'
    verbose = 0  # 0: no output, 1: summary, 2: detailed optimization logs
    adaptive_window_gauss_fit = True  # Whether to use adaptive Gaussian fitting
    plot_gausian_heatmaps = False  # Whether to plot heatmaps with Gaussian fits for each patch
    img_A_shape = (224, 224)  # Original image size for corner error computation
    img_B_shape = (896, 896)  # Original image size for corner error computation
    sample_nr = 128
    sample_index = 16
    tensors_dir = "./tensors"

    gauss_in = dict(cfg.get("gausian_config", cfg.get("gaussian_config", {})))
    gauss_in.setdefault("adaptive_gauss_fit", adaptive_window_gauss_fit)
    gauss_cfg = gaussian_config(gauss_in)

    opt_in = dict(cfg.get("optimize_param", cfg.get("optimize_params", {})))
    opt_in.setdefault("model", mode)
    opt_cfg = optimize_params(opt_in, quiet=verbose == 0)

    gt = torch.load(f"{tensors_dir}/input_sample_{sample_nr:06d}.pt", map_location=torch.device("cpu"))

    H_gt = gt["homography_gt"]
    im_A = gt["im_A"]  # tensor [C, H, W] 224, 224
    im_B = gt["im_B"]  # tensor [C, H, W] 896, 896

    # IN data
    sample = torch.load(f"{tensors_dir}/sample_{sample_nr:03d}_tensor.pt", map_location=torch.device("cpu"))
    loggits = sample[sample_index]["gm_cls"][0]  # shape: (out*out, in, in), e.g. (4096, 14, 14)

    print(f"Logits shape: {loggits.shape}, dtype: {loggits.dtype}")
    in_patch_dim = loggits.shape[-1]
    out_patch_dim = int(loggits.shape[0] ** 0.5)
    assert out_patch_dim * out_patch_dim == loggits.shape[0]

    pts_A, means_B, peaks_B, covs_B = find_gaussians(
        loggits,
        plot_heatmaps=plot_gausian_heatmaps,
        adaptive_gauss_fit=gauss_cfg["adaptive_gauss_fit"],
        plotter=plot_heatmap_comparison,
        adaptive_threshold=gauss_cfg["adaptive_threshold"],
        adaptive_n_sigma=gauss_cfg["adaptive_n_sigma"],
        adaptive_max_iter=gauss_cfg["adaptive_max_iter"],
        adaptive_min_half_w=gauss_cfg["adaptive_min_half_w"],
        adaptive_max_half_w=gauss_cfg["adaptive_max_half_w"],
        fixed_threshold=gauss_cfg["fixed_threshold"],
        fixed_window_size=gauss_cfg["fixed_window_size"],
    )

    print(f"Extracted {pts_A.shape[0]} Gaussian correspondences:")

    H_final, Hinit = optimize_homography(
        pts_A,
        means_B,
        covs_B,
        peaks_B=peaks_B,
        verbose=verbose,
        **opt_cfg,
    )

    # Convert estimated homographies to dataloader space
    H_final_dl = convert_to_dataloader_homography(H_final, 14, 64, crop_res=img_A_shape, map_res=img_B_shape)
    Hinit_dl = convert_to_dataloader_homography(Hinit, 14, 64, crop_res=img_A_shape, map_res=img_B_shape)

    # Homography from image A pixels to image B pixels
    H_A_to_B = convert_to_pixel_homography(
        H_final,
        in_patch_dim,
        out_patch_dim,
        crop_res=img_A_shape,
        map_res=img_B_shape,
    )

    print("\n--- Ground Truth Homography ---")
    H_gt_np = H_gt.numpy() if hasattr(H_gt, "numpy") else H_gt
    print(np.round(H_gt_np, 4))

    print("\n--- Final Optimized Homography Matrix (Dataloader Space) ---")
    print(np.round(H_final_dl, 4))
    err_final = compute_corner_error(H_gt_np, H_final_dl, w=img_A_shape[0], h=img_A_shape[1])
    print(f"Corner Error (Final) vs GT: {err_final:.4f} pixels")

    print("\n--- Initial Homography Matrix from RANSAC (Dataloader Space) ---")
    print(np.round(Hinit_dl, 4))
    err_init = compute_corner_error(H_gt_np, Hinit_dl, w=img_A_shape[0], h=img_A_shape[1])
    print(f"Corner Error (Init) vs GT: {err_init:.4f} pixels")

    plot_correspondences_with_arrows(
        pts_A,
        means_B,
        peaks_B,
        covs_B,
        size_a=in_patch_dim,
        size_b=out_patch_dim,
    )

    plot_homography_projection(
        pts_A,
        means_B,
        H_final,
        size_a=in_patch_dim,
        size_b=out_patch_dim,
        title_suffix="Final Optimized H",
    )
    plot_homography_projection(
        pts_A,
        peaks_B,
        Hinit,
        size_a=in_patch_dim,
        size_b=out_patch_dim,
        title_suffix="Initial RANSAC H",
    )

    # Image A onto Image B using final dataloader space homography
    plot_image_homography_warp(im_A, im_B, H_A_to_B)

    plt.show()
