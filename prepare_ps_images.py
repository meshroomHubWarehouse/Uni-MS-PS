#!/usr/bin/env python3
"""
Prepare Images for Photometric Stereo - Standalone Script

This script preprocesses images for photometric stereo processing by:
1. Organizing images by pose for photometric stereo algorithms
2. Extracting and processing masks from alpha channels or external files
3. Optionally ensuring landscape orientation
4. Optionally cropping images based on mask bounding boxes
5. Creating the data structure expected by photometric stereo algorithms

The output includes both a preprocessed SfMData file and organized image folders
ready for photometric stereo processing.

This is a standalone version of the Meshroom PreparePSImages node.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Try to import pyalicevision for SfMData handling
try:
    from pyalicevision import sfmData, sfmDataIO, camera, numeric
    HAS_PYALICEVISION = True
except ImportError:
    HAS_PYALICEVISION = False
    print("Warning: pyalicevision not found. SfMData support will be disabled.")


def setup_logging(verbose_level: str) -> logging.Logger:
    """Setup logging with the specified verbosity level."""
    level_map = {
        "fatal": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "trace": logging.DEBUG,  # Python doesn't have TRACE, use DEBUG
    }
    
    level = level_map.get(verbose_level.lower(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging.getLogger(__name__)


def ensure_landscape_orientation(image):
    """
    Ensure image is in landscape orientation (width > height).
    Returns the image and rotation info.
    """
    height, width = image.shape[:2]
    
    if height > width:
        # Rotate 90 degrees clockwise to make it landscape
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotation_applied = 90
    else:
        # Already landscape or square
        rotated_image = image
        rotation_applied = 0
    
    return rotated_image, rotation_applied


def estimate_common_bbox_from_masks(mask_paths, target_size=None):
    """
    Estimate a common bounding box from multiple mask files.
    
    Args:
        mask_paths: List of paths to mask files
        target_size: Optional tuple (width, height) to constrain the bbox
        
    Returns:
        bbox: (x, y, width, height) or None if no valid masks found
    """
    if not mask_paths:
        return None
    
    valid_masks = []
    original_size = None
    
    for mask_path in mask_paths:
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Ensure landscape orientation for mask too
        mask, _ = ensure_landscape_orientation(mask)
        
        if original_size is None:
            original_size = (mask.shape[1], mask.shape[0])  # (width, height)
        elif (mask.shape[1], mask.shape[0]) != original_size:
            # Resize mask to match the first valid mask size
            mask = cv2.resize(mask, original_size)
        
        valid_masks.append(mask)
    
    if not valid_masks:
        return None
    
    # Combine all masks (union)
    combined_mask = np.zeros_like(valid_masks[0])
    for mask in valid_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Find bounding box of the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get bounding box of all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = -float('inf'), -float('inf')
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    if x_min == float('inf'):
        return None
    
    bbox_width = int(x_max - x_min)
    bbox_height = int(y_max - y_min)
    
    # Apply target size constraints if specified
    if target_size:
        target_width, target_height = target_size
        
        # Center the bbox and adjust to target size
        center_x = x_min + bbox_width // 2
        center_y = y_min + bbox_height // 2
        
        x_min = max(0, center_x - target_width // 2)
        y_min = max(0, center_y - target_height // 2)
        
        # Ensure we don't exceed image boundaries
        x_min = min(x_min, original_size[0] - target_width)
        y_min = min(y_min, original_size[1] - target_height)
        
        bbox_width = target_width
        bbox_height = target_height
    
    return (int(x_min), int(y_min), int(bbox_width), int(bbox_height))


def update_intrinsics_after_rotation_and_crop(intrinsic, rotation_applied, crop_bbox, resize_factor=1.0):
    """
    Update camera intrinsics after rotation, cropping, and resizing.
    
    Args:
        intrinsic: Camera intrinsic object
        rotation_applied: Rotation angle applied (0 or 90 degrees)
        crop_bbox: (x, y, width, height) of the crop
        resize_factor: Factor by which the image is resized (e.g., 0.5 = half size)
        
    Returns:
        Updated intrinsic parameters
    """
    if not HAS_PYALICEVISION:
        return intrinsic
    
    # Cast to Pinhole camera model
    cam = camera.Pinhole.cast(intrinsic)
    if cam is None:
        return intrinsic  # Can't update non-pinhole models
    
    # Apply crop offset
    if crop_bbox:
        crop_x, crop_y, new_width, new_height = crop_bbox
    
        # Get current principal point
        pp = cam.getPrincipalPoint()
        pp_x = numeric.getX(pp)
        pp_y = numeric.getY(pp)
        logging.debug(f"Original principal point: ({pp_x}, {pp_y})")

        # Adjust principal point for crop
        new_ppx = pp_x - crop_x
        new_ppy = pp_y - crop_y
        logging.debug(f"Adjusted principal point after crop: ({new_ppx}, {new_ppy})")

        # Apply resize factor to principal point and dimensions
        if resize_factor != 1.0:
            new_ppx *= resize_factor
            new_ppy *= resize_factor
            new_width = int(new_width * resize_factor)
            new_height = int(new_height * resize_factor)
            logging.debug(f"Adjusted principal point after resize: ({new_ppx}, {new_ppy})")
            logging.debug(f"New dimensions after resize: {new_width}x{new_height}")
            
            # Update focal length (scale by resize factor)
            focal_x = cam.getFocalLengthPixX()
            focal_y = cam.getFocalLengthPixY()
            cam.setScale(np.array([focal_x * resize_factor, focal_y * resize_factor]))
            logging.debug(f"Updated focal length: ({focal_x * resize_factor}, {focal_y * resize_factor})")

        # Get new offset
        new_px = new_ppx - new_width / 2
        new_py = new_ppy - new_height / 2
        logging.debug(f"New offset: ({new_px}, {new_py})")
        
        # Set new offset
        cam.setOffset(np.array([new_px, new_py]))

        # Set new image size
        cam.setWidth(new_width)
        cam.setHeight(new_height)
    
    elif resize_factor != 1.0:
        # No crop but resize is applied
        current_width = cam.getWidth()
        current_height = cam.getHeight()
        new_width = int(current_width * resize_factor)
        new_height = int(current_height * resize_factor)
        
        # Get current principal point
        pp = cam.getPrincipalPoint()
        pp_x = numeric.getX(pp)
        pp_y = numeric.getY(pp)
        
        # Scale principal point
        new_ppx = pp_x * resize_factor
        new_ppy = pp_y * resize_factor
        
        # Update focal length
        focal_x = cam.getFocalLengthPixX()
        focal_y = cam.getFocalLengthPixY()
        cam.setScale(np.array([focal_x * resize_factor, focal_y * resize_factor]))
        logging.debug(f"Updated focal length: ({focal_x * resize_factor}, {focal_y * resize_factor})")
        
        # Get new offset
        new_px = new_ppx - new_width / 2
        new_py = new_ppy - new_height / 2
        
        # Set new offset
        cam.setOffset(np.array([new_px, new_py]))
        
        # Set new image size
        cam.setWidth(new_width)
        cam.setHeight(new_height)
        logging.debug(f"Updated dimensions after resize: {new_width}x{new_height}")
        
    return intrinsic


def prepare_ps_images(
    input_path: str,
    output_folder: str,
    mask_path: str = "",
    enable_landscape_rotation: bool = True,
    enable_cropping: bool = False,
    target_crop_width: int = 0,
    target_crop_height: int = 0,
    resize_factor: float = 1.0,
    data_folder_suffix: str = ".data",
    image_prefix: str = "L",
    verbose_level: str = "info"
):
    """
    Main function to prepare images for photometric stereo processing.
    
    Args:
        input_path: Input SfMData file containing images and poses
        output_folder: Output folder for processed data
        mask_path: Path to a folder containing masks or to a single mask file
        enable_landscape_rotation: Rotate images to landscape orientation
        enable_cropping: Enable cropping based on mask bounding box estimation
        target_crop_width: Target width for cropping (0 = automatic)
        target_crop_height: Target height for cropping (0 = automatic)
        resize_factor: Factor to resize images after cropping (e.g., 0.5 = half size)
        data_folder_suffix: Suffix for data folders (e.g., '.data')
        image_prefix: Prefix for organized images (e.g., 'L')
        verbose_level: Verbosity level
    """
    logger = setup_logging(verbose_level)
    
    if not input_path:
        raise RuntimeError("No input SfMData file provided")
    
    if not HAS_PYALICEVISION:
        raise RuntimeError("pyalicevision is required but not installed")
    
    # Load input SfMData
    input_sfm_data = sfmData.SfMData()
    if not sfmDataIO.load(input_sfm_data, input_path, sfmDataIO.ALL):
        raise RuntimeError(f"Failed to load input SfMData file: {input_path}")
    
    logger.info(f"Loaded SfMData from: {input_path}")
    
    # Create output directories
    output_folder = Path(output_folder)
    output_data_dir = output_folder / "ps_data"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    output_mask_dir = output_folder / "masks"
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    output_sfm_path = output_folder / "sfmData.sfm"
    
    # Collect mask paths for preprocessing if cropping is enabled
    mask_paths = []
    alpha_mask_paths = []
    common_bbox = None
    
    if enable_cropping:
        # Collect external mask files if provided
        if mask_path:
            mask_path_obj = Path(mask_path)
            
            if mask_path_obj.is_dir():
                # Look for mask files in directory
                for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    mask_paths.extend(list(mask_path_obj.glob(f'*{ext}')))
                    mask_paths.extend(list(mask_path_obj.glob(f'*{ext.upper()}')))
            elif mask_path_obj.is_file():
                # Single mask file
                mask_paths = [mask_path_obj]
        
        # Collect alpha channels ONLY from representative images (viewId == poseId)
        views = input_sfm_data.getViews()
        
        for view_id, view in views.items():
            pose_id = view.getPoseId()
            
            # Only process representative images (viewId == poseId)
            if view_id != pose_id:
                continue
            
            image_path = view.getImage().getImagePath()
            
            if os.path.exists(image_path):
                # Check if image has alpha channel
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    # Extract alpha channel and save as temporary mask for preprocessing
                    alpha_mask = img[:, :, 3]
                    temp_mask_path = output_data_dir / f"temp_alpha_mask_{pose_id}.png"
                    temp_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(temp_mask_path), alpha_mask)
                    alpha_mask_paths.append(temp_mask_path)
                    logger.debug(f"Extracted alpha channel from representative image for pose {pose_id}")
        
        # Combine external masks and alpha channel masks
        all_mask_paths = mask_paths + alpha_mask_paths
        
        if all_mask_paths:
            logger.info(f"Collected {len(mask_paths)} external masks and {len(alpha_mask_paths)} alpha channel masks for preprocessing")
            
            # Determine target crop size
            target_crop_size = None
            if target_crop_width > 0 and target_crop_height > 0:
                target_crop_size = (target_crop_width, target_crop_height)
            
            # Estimate common bbox if cropping is enabled and masks are provided
            logger.info(f"Estimating common bbox from {len(all_mask_paths)} masks")
            common_bbox = estimate_common_bbox_from_masks(all_mask_paths, target_crop_size)
            if common_bbox:
                logger.info(f"Common bbox estimated: {common_bbox}")
            else:
                logger.warning("Could not estimate common bbox from masks")
        else:
            logger.info("No masks found for cropping")

        # Clean up temporary alpha mask files
        for temp_mask_path in alpha_mask_paths:
            try:
                os.remove(temp_mask_path)
            except OSError:
                pass  # Ignore cleanup errors
    else:
        logger.info("Cropping disabled")
    
    if resize_factor != 1.0:
        logger.info(f"Resize factor: {resize_factor}")
    
    # Get views per pose ID
    views_per_pose_id = {}
    views = input_sfm_data.getViews()
    for view_id, view in views.items():
        pose_id = view.getPoseId()
        if pose_id not in views_per_pose_id:
            views_per_pose_id[pose_id] = []
        views_per_pose_id[pose_id].append(view_id)
    
    # Create updated SfMData with new image paths
    updated_sfm_data = input_sfm_data
    processed_count = 0
    updated_intrinsics = set()  # Track which intrinsics have been updated
    
    # Process each pose separately
    for pose_id, view_ids in views_per_pose_id.items():
        logger.info(f"Processing Pose ID: {pose_id}")
        
        # Create pose directory
        pose_dir = output_data_dir / f"pose_{pose_id}{data_folder_suffix}"
        pose_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image list for this pose
        image_list = []
        for view_id in view_ids:
            view = input_sfm_data.getView(view_id)
            image_path = view.getImage().getImagePath()
            image_list.append((view_id, image_path))
        
        if len(image_list) < 1:
            logger.warning(f"Empty image list for pose {pose_id}, skipping.")
            continue
        
        # Process images with photometric stereo naming
        prefix = image_prefix
        alpha_mask = None
        alpha_mask_found = False
        rotation_applied = 0
        original_size = None
        
        for i, (view_id, image_path) in enumerate(image_list):
            if not os.path.isfile(image_path):
                logger.warning(f"Image file not found: {image_path}")
                continue
            
            src_path = Path(image_path)
            # Create filename with prefix and zero-padded index
            dst_filename = f"{prefix}{i:03d}{src_path.suffix}"
            dst_path = pose_dir / dst_filename
            
            # Load and preprocess image
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                continue
            
            # Store original size for intrinsic update (only for first representative image)
            if view_id == pose_id and original_size is None:
                if len(img.shape) == 3 and img.shape[2] == 4:
                    original_size = (img.shape[1], img.shape[0])  # (width, height) without alpha
                else:
                    original_size = (img.shape[1], img.shape[0])  # (width, height)
            
            # Handle alpha channel extraction - ONLY from representative image (viewId == poseId)
            if view_id == pose_id and len(img.shape) == 3 and img.shape[2] == 4:
                logger.debug(f"Alpha channel detected in representative image {src_path.name}")
                
                # Extract alpha channel as mask from representative image
                alpha_mask = img[:, :, 3]  # Extract alpha channel
                alpha_mask_found = True
                logger.debug(f"Using alpha channel from representative image {src_path.name} as mask for pose {pose_id}")
            
            # Remove alpha channel from all images that have it
            if len(img.shape) == 3 and img.shape[2] == 4:
                # Remove alpha channel and keep only RGB
                img = img[:, :, :3]  # Keep only RGB channels
            
            # Apply preprocessing: ensure landscape orientation (simple rotation, no intrinsic update)
            processed_img = img
            if enable_landscape_rotation:
                processed_img, current_rotation = ensure_landscape_orientation(img)
                if view_id == pose_id:  # Store rotation info from representative image
                    rotation_applied = current_rotation
                if current_rotation > 0:
                    logger.debug(f"Rotated image {src_path.name} by {current_rotation} degrees")
            
            # Apply cropping if enabled and bbox is available
            if enable_cropping and common_bbox:
                crop_x, crop_y, crop_w, crop_h = common_bbox
                processed_img = processed_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                logger.debug(f"Cropped image {src_path.name} to {crop_w}x{crop_h}")
            
            # Apply resize if factor is not 1.0
            if resize_factor != 1.0:
                new_h = int(processed_img.shape[0] * resize_factor)
                new_w = int(processed_img.shape[1] * resize_factor)
                # Use INTER_NEAREST to preserve exact pixel values (no interpolation/averaging)
                processed_img = cv2.resize(processed_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                logger.debug(f"Resized image {src_path.name} to {new_w}x{new_h}")
            
            # Save processed image
            cv2.imwrite(str(dst_path), processed_img)
            logger.debug(f"Processed and saved {src_path.name} -> {dst_filename}")
            
            # Update SfMData with new image path and dimensions
            view = updated_sfm_data.getView(view_id)
            view.getImage().setImagePath(str(dst_path))
            
            # Update image dimensions if cropping or resizing was applied
            if (enable_cropping and common_bbox) or resize_factor != 1.0:
                new_height, new_width = processed_img.shape[:2]
                view.getImage().setWidth(new_width)
                view.getImage().setHeight(new_height)
                logger.debug(f"Updated image dimensions for {src_path.name}: {new_width}x{new_height}")
            
            processed_count += 1
        
        # Handle mask: alpha channel takes priority, then external mask
        mask_copied = False
        
        # First, process and save alpha channel mask if found
        if alpha_mask_found and alpha_mask is not None:
            # Apply same preprocessing to alpha mask
            processed_mask = alpha_mask
            if enable_landscape_rotation and rotation_applied > 0:
                processed_mask, _ = ensure_landscape_orientation(alpha_mask)
            if enable_cropping and common_bbox:
                crop_x, crop_y, crop_w, crop_h = common_bbox
                processed_mask = processed_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # Apply resize to mask if factor is not 1.0
            if resize_factor != 1.0:
                new_h = int(processed_mask.shape[0] * resize_factor)
                new_w = int(processed_mask.shape[1] * resize_factor)
                processed_mask = cv2.resize(processed_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                logger.debug(f"Resized alpha mask for pose {pose_id} to {new_w}x{new_h}")
            
            # Save mask in pose directory
            mask_file_path = pose_dir / "mask.png"
            cv2.imwrite(str(mask_file_path), processed_mask)
            logger.debug(f"Saved processed alpha channel mask for pose {pose_id}")
            
            # Save mask in output mask folder with pose ID naming
            mask_output_path = output_mask_dir / f"{pose_id}.png"
            cv2.imwrite(str(mask_output_path), processed_mask)
            logger.debug(f"Saved mask to mask folder: {mask_output_path}")
            
            mask_copied = True
        
        # If no alpha mask and external mask path provided, try external masks
        elif mask_path:
            mask_to_process = None
            mask_source = None
            
            if os.path.isdir(mask_path):
                # Look for pose-specific mask
                mask_patterns = [
                    f"pose_{pose_id}_mask.png",
                    f"{pose_id}_mask.png", 
                    f"mask_{pose_id}.png",
                    f"{pose_id}.png",
                    "mask.png"  # Generic mask
                ]
                for pattern in mask_patterns:
                    mask_file = os.path.join(mask_path, pattern)
                    if os.path.isfile(mask_file):
                        mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            mask_to_process = mask_img
                            mask_source = pattern
                            break
                
                if mask_to_process is None:
                    logger.debug(f"No external mask found for pose {pose_id} in {mask_path}")
                    
            elif os.path.isfile(mask_path):
                # Single mask file for all poses
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_to_process = mask_img
                    mask_source = "global external mask"
            else:
                logger.debug(f"External mask path is neither a directory nor a file: {mask_path}")
            
            # Process the external mask if found
            if mask_to_process is not None:
                processed_mask = mask_to_process
                
                # Apply the same preprocessing transformations as the images
                if enable_landscape_rotation and rotation_applied > 0:
                    processed_mask, _ = ensure_landscape_orientation(mask_to_process)
                    logger.debug(f"Rotated mask for pose {pose_id} by {rotation_applied} degrees")
                
                if enable_cropping and common_bbox:
                    crop_x, crop_y, crop_w, crop_h = common_bbox
                    processed_mask = processed_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    logger.debug(f"Cropped mask for pose {pose_id} to {crop_w}x{crop_h}")
                
                # Apply resize to mask if factor is not 1.0
                if resize_factor != 1.0:
                    new_h = int(processed_mask.shape[0] * resize_factor)
                    new_w = int(processed_mask.shape[1] * resize_factor)
                    processed_mask = cv2.resize(processed_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    logger.debug(f"Resized external mask for pose {pose_id} to {new_w}x{new_h}")
                
                # Save mask in pose directory
                mask_file_path = pose_dir / "mask.png"
                cv2.imwrite(str(mask_file_path), processed_mask)
                logger.debug(f"Copied and processed {mask_source} for pose {pose_id}")
                
                # Save mask in output mask folder with pose ID naming
                mask_output_path = output_mask_dir / f"{pose_id}.png"
                cv2.imwrite(str(mask_output_path), processed_mask)
                logger.debug(f"Saved mask to mask folder: {mask_output_path}")
                
                mask_copied = True

        if not mask_copied:
            logger.debug(f"No mask used for pose {pose_id}")

        logger.info(f"Prepared {len(image_list)} images for pose {pose_id} in {pose_dir}")

        # Update intrinsics only once per intrinsic_id (after processing all images for this pose)
        if ((enable_cropping and common_bbox) or resize_factor != 1.0) and original_size:
            # Get representative view for this pose
            representative_view = updated_sfm_data.getView(pose_id)
            intrinsic_id = representative_view.getIntrinsicId()
            
            # Only update if this intrinsic hasn't been updated yet
            if intrinsic_id not in updated_intrinsics:
                if updated_sfm_data.getIntrinsics().count(intrinsic_id) > 0:
                    intrinsic = updated_sfm_data.getIntrinsics()[intrinsic_id]
                    
                    # Update intrinsics considering rotation, crop, and resize
                    updated_intrinsic = update_intrinsics_after_rotation_and_crop(
                        intrinsic, rotation_applied, common_bbox, resize_factor
                    )
                    
                    # Mark this intrinsic as updated
                    updated_intrinsics.add(intrinsic_id)
                    logger.info(f"Updated intrinsics for intrinsic_id {intrinsic_id} (used by pose {pose_id})")
                else:
                    logger.warning(f"Intrinsic {intrinsic_id} not found for pose {pose_id}")

    # Save updated SfMData
    if not sfmDataIO.save(updated_sfm_data, str(output_sfm_path), sfmDataIO.ALL):
        logger.warning(f"Failed to save updated SfMData to: {output_sfm_path}")
    else:
        logger.info(f"Updated SfMData saved to: {output_sfm_path}")

    logger.info(f"Photometric stereo data preparation completed")
    logger.info(f"Processed {processed_count} images total")
    logger.info(f"Output data folder: {output_data_dir}")
    logger.info(f"Output mask folder: {output_mask_dir}")
    
    return {
        "output_sfm_data": str(output_sfm_path),
        "output_data_folder": str(output_data_dir),
        "output_mask_folder": str(output_mask_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Images for Photometric Stereo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input sfmData.sfm --output ./output
  %(prog)s --input sfmData.sfm --output ./output --mask-path ./masks --enable-cropping
  %(prog)s --input sfmData.sfm --output ./output --no-landscape-rotation
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        dest="input_path",
        required=True,
        help="Input SfMData file containing images and poses."
    )
    
    parser.add_argument(
        "--output", "-o",
        dest="output_folder",
        required=True,
        help="Output folder for processed data."
    )
    
    # Optional arguments
    parser.add_argument(
        "--mask-path", "-m",
        dest="mask_path",
        default="",
        help="Path to a folder containing masks or to a single mask file."
    )
    
    parser.add_argument(
        "--enable-landscape-rotation",
        dest="enable_landscape_rotation",
        action="store_true",
        default=True,
        help="Rotate images to landscape orientation (width > height). Default: enabled."
    )
    
    parser.add_argument(
        "--no-landscape-rotation",
        dest="enable_landscape_rotation",
        action="store_false",
        help="Disable landscape rotation."
    )
    
    parser.add_argument(
        "--enable-cropping",
        dest="enable_cropping",
        action="store_true",
        default=False,
        help="Enable cropping based on mask bounding box estimation. Default: disabled."
    )
    
    parser.add_argument(
        "--target-crop-width",
        dest="target_crop_width",
        type=int,
        default=0,
        help="Target width for cropping (0 = automatic based on masks). Default: 0."
    )
    
    parser.add_argument(
        "--target-crop-height",
        dest="target_crop_height",
        type=int,
        default=0,
        help="Target height for cropping (0 = automatic based on masks). Default: 0."
    )
    
    parser.add_argument(
        "--resize-factor",
        dest="resize_factor",
        type=float,
        default=1.0,
        help="Factor to resize images after cropping (e.g., 0.5 = half size, 2.0 = double size). Default: 1.0 (no resize)."
    )
    
    parser.add_argument(
        "--data-folder-suffix",
        dest="data_folder_suffix",
        default=".data",
        help="Suffix for data folders (e.g., '.data' for pose_0.data). Default: '.data'."
    )
    
    parser.add_argument(
        "--image-prefix",
        dest="image_prefix",
        default="L",
        help="Prefix for organized images (e.g., 'L' for L000.jpg, L001.jpg). Default: 'L'."
    )
    
    parser.add_argument(
        "--verbose", "-v",
        dest="verbose_level",
        choices=["fatal", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Verbosity level. Default: info."
    )
    
    args = parser.parse_args()
    
    try:
        result = prepare_ps_images(
            input_path=args.input_path,
            output_folder=args.output_folder,
            mask_path=args.mask_path,
            enable_landscape_rotation=args.enable_landscape_rotation,
            enable_cropping=args.enable_cropping,
            target_crop_width=args.target_crop_width,
            target_crop_height=args.target_crop_height,
            resize_factor=args.resize_factor,
            data_folder_suffix=args.data_folder_suffix,
            image_prefix=args.image_prefix,
            verbose_level=args.verbose_level
        )
        
        print(f"\nOutput files:")
        print(f"  SfMData:     {result['output_sfm_data']}")
        print(f"  Data folder: {result['output_data_folder']}")
        print(f"  Mask folder: {result['output_mask_folder']}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
