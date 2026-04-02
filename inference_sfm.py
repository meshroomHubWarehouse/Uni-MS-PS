"""
SfM-based inference for Uni-MS-PS.

Reads an AliceVision SfMData JSON, runs photometric stereo per pose,
and produces an output JSON mapping poseIds to normal map paths.

The crop/uncrop around masks is handled internally — output normal
maps are full-resolution (or downscaled) images, not cropped patches.
"""

import argparse
import copy
import json
import logging
import os
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sfm_loader import load_sfm, group_views_by_pose, load_imgs_mask_sfm
from utils import load_model, process_normal, depadding, normal_to_rgb_16bits


logger = logging.getLogger(__name__)


def uncrop_normal(normal, crop_bbox, original_shape, is_portrait):
    """Restore a cropped normal map to its original image dimensions.

    Args:
        normal: (H_crop, W_crop, 3) normal map from process_normal
        crop_bbox: (x_min, x_max_pad, y_min, y_max_pad)
        original_shape: (H, W, C) of the mask/image before cropping
        is_portrait: whether the image was transposed to landscape

    Returns:
        Full-size normal map (H_orig, W_orig, 3)
    """
    x_min, x_max_pad, y_min, y_max_pad = crop_bbox

    # Pad back to original size (landscape space)
    pad_x_min = np.zeros((x_min, normal.shape[1], 3))
    pad_x_max = np.zeros((x_max_pad, normal.shape[1], 3))
    normal = np.concatenate((pad_x_min, normal, pad_x_max), axis=0)

    pad_y_min = np.zeros((normal.shape[0], y_min, 3))
    pad_y_max = np.zeros((normal.shape[0], y_max_pad, 3))
    normal = np.concatenate((pad_y_min, normal, pad_y_max), axis=1)

    # Transpose back to portrait if needed
    if is_portrait:
        normal = np.transpose(normal, (1, 0, 2))
        normal = normal[:, :, [1, 0, 2]]  # Swap X and Y normal components

    return normal


def save_normal_exr(normal, out_path):
    """Save normal map as float32 EXR (raw [-1,1] values, BGR for cv2)."""
    cv2.imwrite(out_path, normal[:, :, ::-1].astype(np.float32))


def scale_intrinsics(sfm, downscale):
    """Scale intrinsics and view dimensions to match downscaled images."""
    if downscale <= 1:
        return
    f = float(downscale)
    for intr in sfm.get("intrinsics", []):
        for key in ("width", "height"):
            if key in intr:
                intr[key] = str(int(int(float(str(intr[key]))) / f))
        if "principalPoint" in intr:
            pp = intr["principalPoint"]
            intr["principalPoint"] = [
                str(float(str(pp[0])) / f),
                str(float(str(pp[1])) / f),
            ]
        if "pxFocalLength" in intr:
            pfl = intr["pxFocalLength"]
            if isinstance(pfl, list):
                intr["pxFocalLength"] = [pfl[0] / f, pfl[1] / f]
            else:
                intr["pxFocalLength"] = float(pfl) / f
    for view in sfm.get("views", []):
        for key in ("width", "height"):
            if key in view:
                view[key] = str(int(int(float(str(view[key]))) / f))


def create_output_sfm(sfm_data, output_folder, ext, downscale=1):
    """Create an output SfMData JSON referencing the generated normal maps.

    Filters views to representative ones (viewId == poseId), updates paths
    to point to normal maps, and scales intrinsics for the downscale factor.

    Args:
        sfm_data: input SfMData dict
        output_folder: folder containing the normal maps
        ext: file extension including dot (e.g. ".png", ".exr")
        downscale: integer downscale factor applied to images

    Returns:
        Path to the output SfMData JSON file.
    """
    sfm = copy.deepcopy(sfm_data)

    views = sfm.get("views", [])
    representative_views = []
    for view in views:
        view_id = str(view.get("viewId", ""))
        pose_id = str(view.get("poseId", ""))
        if view_id == pose_id:
            map_path = os.path.join(output_folder,
                                    "{}{}".format(pose_id, ext))
            view["path"] = map_path
            representative_views.append(view)

    sfm["views"] = representative_views
    scale_intrinsics(sfm, downscale)

    output_path = os.path.join(output_folder, "normalMaps.sfm")
    with open(output_path, "w") as f_out:
        json.dump(sfm, f_out, indent=4)

    logger.info("Saved output SfMData to %s", output_path)
    return output_path


def run_sfm_inference(sfm_path, output_folder, mask_folder=None,
                      mask_output_folder=None,
                      nb_img=-1, downscale=1, use_cuda=True,
                      calibrated=False, weights_path="weights",
                      output_format="png16"):
    """Run Uni-MS-PS inference on all poses in an SfM JSON file.

    Args:
        sfm_path: path to input SfMData JSON
        output_folder: where to write normal maps and output JSON
        mask_folder: optional folder with masks (named by poseId or viewId)
        nb_img: number of images per pose (-1 = all)
        downscale: integer downscale factor for input images
        use_cuda: use GPU
        calibrated: use calibrated model
        weights_path: path to model weights directory

    Returns:
        Path to the output JSON file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load SfM data
    sfm_data = load_sfm(sfm_path)
    pose_groups = group_views_by_pose(sfm_data)
    logger.info(f"Loaded {len(sfm_data.get('views', []))} views, "
                f"{len(pose_groups)} poses")

    # Load model once
    logger.info("Loading model...")
    model = load_model(
        path_weight=weights_path,
        cuda=use_cuda,
        mode_inference=True,
        calibrated=calibrated,
    )
    logger.info("Model loaded")

    # Process each pose
    results = []
    total_start = time.time()

    for pose_id, views in tqdm(pose_groups.items(), desc="Poses"):
        logger.info(f"=== Pose {pose_id} ({len(views)} views) ===")
        pose_start = time.time()

        try:
            imgs, mask, padding, crop_bbox, original_shape, is_portrait = \
                load_imgs_mask_sfm(
                    views=views,
                    mask_folder=mask_folder,
                    mask_output_folder=mask_output_folder,
                    nb_img=nb_img,
                    calibrated=calibrated,
                    downscale=downscale,
                )

            # Inference
            normal = process_normal(
                model=model,
                imgs=imgs,
                mask=mask,
                is_portrait=is_portrait,
            )

            # Remove padding from model output
            normal = depadding(normal, padding=padding)

            # Normalize
            normal = torch.from_numpy(normal)
            normal = torch.nn.functional.normalize(normal, 2, -1).numpy()

            # Uncrop to original size
            normal_full = uncrop_normal(
                normal, crop_bbox, original_shape, is_portrait)

            # Save normal map
            ext = ".exr" if output_format == "exr" else ".png"
            out_path = os.path.join(output_folder, f"{pose_id}{ext}")
            if output_format == "exr":
                save_normal_exr(normal_full, out_path)
            else:
                normal_rgb = normal_to_rgb_16bits(normal_full)
                cv2.imwrite(out_path, normal_rgb[:, :, ::-1],
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

            pose_time = time.time() - pose_start
            logger.info(f"Pose {pose_id}: {normal_full.shape}, "
                        f"saved to {out_path} ({pose_time:.1f}s)")

            # Find the representative view (viewId == poseId) for pose info
            rep_view = None
            for v in views:
                if str(v.get("viewId")) == str(pose_id):
                    rep_view = v
                    break
            if rep_view is None:
                rep_view = views[0]

            results.append({
                "poseId": pose_id,
                "viewId": str(rep_view.get("viewId")),
                "normalMapPath": os.path.abspath(out_path),
                "width": normal_full.shape[1],
                "height": normal_full.shape[0],
                "nbImages": len(views),
            })

        except Exception as e:
            logger.error(f"Failed on pose {pose_id}: {e}")
            continue

    total_time = time.time() - total_start
    logger.info(f"All poses processed in {total_time:.1f}s")

    # Generate output SfMData JSON with normal map references
    ext = ".exr" if output_format == "exr" else ".png"
    output_sfm_path = create_output_sfm(
        sfm_data, output_folder, ext, downscale=downscale)

    logger.info(f"Inference complete: {len(results)} poses processed")
    return output_sfm_path


def main():
    parser = argparse.ArgumentParser(
        description="Uni-MS-PS inference from SfM JSON"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input SfMData JSON file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output folder for normal maps")
    parser.add_argument("--masks", "-m", default=None,
                        help="Folder with mask PNGs (named by poseId/viewId)")
    parser.add_argument("--nb-img", type=int, default=-1,
                        help="Number of images per pose (-1 = all)")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Integer downscale factor (1 = no downscale)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use GPU")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use calibrated model")
    parser.add_argument("--weights", default="weights",
                        help="Path to model weights directory")
    parser.add_argument("--output-format", default="png16",
                        choices=["png16", "exr"],
                        help="Output format: png16 (16-bit PNG) or "
                             "exr (float32 EXR) (default: png16)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_sfm = run_sfm_inference(
        sfm_path=args.input,
        output_folder=args.output,
        mask_folder=args.masks,
        nb_img=args.nb_img,
        downscale=args.downscale,
        use_cuda=args.cuda,
        calibrated=args.calibrated,
        weights_path=args.weights,
        output_format=args.output_format,
    )
    print(f"Output SfMData: {output_sfm}")


if __name__ == "__main__":
    main()
