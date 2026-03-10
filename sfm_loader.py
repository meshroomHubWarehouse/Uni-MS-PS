"""
SfM JSON dataloader for Uni-MS-PS.

Reads an AliceVision SfMData JSON file, groups views by poseId
(multi-lighting images sharing the same camera pose), and provides
the same tensor interface as load_imgs_mask for the inference pipeline.
"""

import json
import logging
import os

import cv2
import numpy as np
import torch

from utils import resize_with_padding, get_nb_stage

logger = logging.getLogger(__name__)


def load_sfm(sfm_path):
    """Load SfMData — try pyalicevision first (supports .sfm, .abc, .json),
    fallback to json.load.  Uses sfmDataIO.save() to a temp JSON so ALL
    fields (version, intrinsics, metadata, surveys…) are preserved."""
    try:
        from pyalicevision import sfmData as avSfmData, sfmDataIO
        import tempfile
        data = avSfmData.SfMData()
        if sfmDataIO.load(data, sfm_path, sfmDataIO.ALL):
            logger.info("Loaded SfMData via pyalicevision: %s", sfm_path)
            with tempfile.NamedTemporaryFile(suffix=".sfm", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sfmDataIO.save(data, tmp_path, sfmDataIO.ALL)
                with open(tmp_path, "r") as f:
                    return json.load(f)
            finally:
                os.unlink(tmp_path)
        logger.warning("pyalicevision failed to load %s, falling back to JSON", sfm_path)
    except ImportError:
        logger.info("pyalicevision not available, using JSON loader")
    with open(sfm_path, "r") as f:
        return json.load(f)


load_sfm_json = load_sfm  # backward compat


def group_views_by_pose(sfm_data):
    """Group views by poseId.

    Returns:
        dict mapping poseId (str) -> list of view dicts
    """
    groups = {}
    for view in sfm_data.get("views", []):
        pose_id = str(view.get("poseId", view.get("viewId")))
        groups.setdefault(pose_id, []).append(view)
    return groups


def extract_alpha_mask(views):
    """Extract mask by ANDing all alpha channels from the pose's images.

    Images with all-white alpha are skipped. The result is the intersection
    of all non-trivial alpha masks, keeping only the object area.

    Returns a grayscale numpy array or None.
    """
    combined = None
    count = 0
    for v in views:
        path = v.get("path", "")
        if not path:
            img_obj = v.get("image", {})
            path = img_obj.get("path", img_obj.get("imagePath", ""))
        if not path or not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or len(img.shape) != 3 or img.shape[2] != 4:
            continue
        alpha = img[:, :, 3]
        # Skip all-white (trivial) alpha channels
        if alpha.min() > 250:
            continue
        mask = (alpha > 0).astype(np.uint8)
        if combined is None:
            combined = mask
        else:
            combined = combined * mask  # logical AND
        count += 1
    if combined is not None:
        logger.info("Extracted alpha mask from %d images (AND)", count)
        return (combined * 255).astype(np.uint8)
    return None


def find_mask_for_pose(pose_id, mask_folder, view_ids=None, views=None):
    """Find a mask file for a given pose.

    Search order:
        1. {pose_id}.png in mask_folder
        2. {any viewId in this pose}.png in mask_folder
        3. mask.png (global fallback) in mask_folder
        4. Alpha channel from input images

    Returns:
        numpy array (grayscale) or None
    """
    if mask_folder and os.path.isdir(mask_folder):
        # Try pose_id directly
        candidate = os.path.join(mask_folder, f"{pose_id}.png")
        if os.path.isfile(candidate):
            mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask

        # Try each viewId
        if view_ids:
            for vid in view_ids:
                candidate = os.path.join(mask_folder, f"{vid}.png")
                if os.path.isfile(candidate):
                    mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        return mask

        # Global fallback
        candidate = os.path.join(mask_folder, "mask.png")
        if os.path.isfile(candidate):
            mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask

    # Fallback: extract from alpha channel
    if views:
        return extract_alpha_mask(views)

    return None


def load_imgs_mask_sfm(views, mask_folder=None, mask_output_folder=None,
                       nb_img=-1, calibrated=False, downscale=1,
                       max_size=None):
    """Load images and mask for one pose group from SfM views.

    This is the SfM-aware equivalent of load_imgs_mask. It produces
    the same outputs so that run() / process_normal() work unchanged.

    Args:
        views: list of view dicts (all sharing the same poseId)
        mask_folder: optional folder with mask PNGs named by viewId/poseId
        nb_img: number of images to use (-1 = all)
        calibrated: whether to load light directions (not used for SfM)
        downscale: integer downscale factor (1 = no downscale, 2 = half, etc.)
        max_size: optional max dimension

    Returns:
        Same tuple as load_imgs_mask:
        (imgs_tensor, mask_tensor, padding, zoom_coord, original_shape,
         is_portrait, crop_bbox)
        crop_bbox is (x_min, x_max_pad, y_min, y_max_pad) in the
        (possibly downscaled) image space — needed to uncrop the output.
    """
    pose_id = str(views[0].get("poseId", views[0].get("viewId")))
    view_ids = [str(v["viewId"]) for v in views]

    # Collect image paths
    image_paths = []
    for v in views:
        path = v.get("path", "")
        if not path:
            # Try nested structure
            img_obj = v.get("image", {})
            path = img_obj.get("path", img_obj.get("imagePath", ""))
        if path and os.path.isfile(path):
            image_paths.append(path)
        else:
            logger.warning(f"Image not found for viewId {v.get('viewId')}: {path}")

    if not image_paths:
        raise RuntimeError(f"No valid images found for pose {pose_id}")

    logger.info(f"Pose {pose_id}: {len(image_paths)} images")

    # Load mask
    mask_img = find_mask_for_pose(pose_id, mask_folder, view_ids, views=views)

    if mask_img is None:
        # Create default mask from first image
        example = cv2.imread(image_paths[0])
        if example is None:
            raise RuntimeError(f"Cannot read first image: {image_paths[0]}")
        if len(example.shape) == 3 and example.shape[2] == 4:
            example = example[:, :, :3]
        if downscale > 1:
            h, w = example.shape[:2]
            example = cv2.resize(example, (w // downscale, h // downscale),
                                 interpolation=cv2.INTER_AREA)
        mask = np.ones(example.shape[:2] + (3,), dtype=np.uint8)
        logger.info(f"No mask found for pose {pose_id}, using full image")
    else:
        mask = mask_img
        if downscale > 1:
            h, w = mask.shape[:2]
            mask = cv2.resize(mask, (w // downscale, h // downscale),
                              interpolation=cv2.INTER_NEAREST)
        # Convert to 3-channel if grayscale
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=-1)
        elif mask.shape[2] == 4:
            mask = mask[:, :, :3]

    # Save extracted mask at output resolution (after downscale)
    if mask_img is not None and mask_output_folder and not mask_folder:
        os.makedirs(mask_output_folder, exist_ok=True)
        mask_path = os.path.join(mask_output_folder, f"{pose_id}.png")
        cv2.imwrite(mask_path, mask)
        logger.info("Saved mask to %s", mask_path)

    # Portrait detection
    is_portrait = mask.shape[0] > mask.shape[1]
    if is_portrait:
        logger.info("Portrait orientation detected, transposing to landscape")
        mask = np.transpose(mask, (1, 0, 2))

    if max_size is not None:
        if mask.shape[0] > max_size or mask.shape[1] > max_size:
            mask = cv2.resize(mask, (max_size, max_size))

    original_shape = mask.shape

    # Crop around mask (same logic as original load_imgs_mask)
    coord = np.argwhere(mask[:, :, 0] > 0)
    if len(coord) == 0:
        # Empty mask — use full image
        x_min, x_max = 0, mask.shape[0]
        y_min, y_max = 0, mask.shape[1]
    else:
        x_min, x_max = np.min(coord[:, 0]), np.max(coord[:, 0])
        y_min, y_max = np.min(coord[:, 1]), np.max(coord[:, 1])

    x_max_pad = mask.shape[0] - x_max
    y_max_pad = mask.shape[1] - y_max
    crop_bbox = (x_min, x_max_pad, y_min, y_max_pad)

    mask = mask[x_min:x_max, y_min:y_max]

    nb_stage = get_nb_stage(mask.shape)
    size_img = 32 * 2 ** (nb_stage - 1)

    logger.info(f"Crop: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}], "
                f"stages={nb_stage}, target_size={size_img}")

    mask, _ = resize_with_padding(mask, expected_size=(size_img, size_img))
    mask = (mask > 0)
    mask = mask[:, :, 0]

    # Select images
    if nb_img is None or nb_img >= len(image_paths) or nb_img == -1:
        selected_paths = image_paths
    else:
        indices = np.random.choice(len(image_paths), nb_img, replace=False)
        selected_paths = [image_paths[i] for i in sorted(indices)]

    logger.info(f"Processing {len(selected_paths)} images")

    # Load and preprocess images
    imgs = []
    padding = None
    for img_path in selected_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue

        # Handle channels
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=-1)

        # Downscale
        if downscale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downscale, h // downscale),
                             interpolation=cv2.INTER_AREA)

        # Portrait transpose
        if is_portrait:
            img = np.transpose(img, (1, 0, 2))

        if max_size is not None:
            if img.shape[0] > max_size or img.shape[1] > max_size:
                img = cv2.resize(img, (max_size, max_size))

        # Crop
        img = img[x_min:x_max, y_min:y_max]

        # Pad to target size
        img, padding = resize_with_padding(img, expected_size=(size_img, size_img))

        img = img.astype(np.float32)
        mean_img = np.mean(img, -1).flatten()
        mean_masked = np.mean(mean_img[mask.flatten()])
        if mean_masked > 0:
            img = img / mean_masked

        imgs.append(img)

    if not imgs:
        raise RuntimeError(f"No images could be loaded for pose {pose_id}")

    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs, -1, 0)
    imgs = torch.from_numpy(imgs).unsqueeze(0).float()

    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    logger.info(f"Image tensor shape: {imgs.shape}")
    logger.info(f"Mask tensor shape: {mask.shape}")

    return imgs, mask, padding, crop_bbox, original_shape, is_portrait
