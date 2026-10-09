"""Meshroom API of Uni-MS-PS: normal map of one multi-lighting pose.

Used by the mrUniMSPS Meshroom plugin, whose common layer (psCommon.py) handles everything else (SfMData,
image selection and loading, masks, outputs). Contract shared by the photometric stereo plugins:

    predictor = loadModel(weightsPath, useGpu, logger)
    maps = predict(predictor, images, mask, **options)

- images: list of float32 RGB arrays (H x W x 3), values as stored in the images (typically [0, 1]),
- mask: bool array (H x W), the pixels of the object,
- maps["normal"]: float32 (H x W x 3) unit normals in the OpenGL camera frame (x right, y up, z towards the
  camera), zero outside the mask and where undefined.

The pre/post-processing is the one of the original Uni-MS-PS inference (utils.load_imgs_mask / run.py): BGR
channel order, crop around the mask, per-image division by the mean intensity over the mask, centered zero
padding to a square of side 32 * 2^k (k + 1 resolution stages), depadding and uncrop. Unlike the original code,
the images are kept as float32 (no 8-bit conversion), the bounding box of the mask is inclusive and portrait
images need no special handling (the network input is square).
"""
import logging
import os

import numpy as np
import torch

# Side of the network input at the coarsest resolution stage: each stage doubles it.
BASE_SIZE = 32

# Native frame of the network output -> OpenGL camera frame (x right, y up, z towards the camera):
# opengl[..., i] = NATIVE_TO_OPENGL_SIGNS[i] * native[..., NATIVE_TO_OPENGL_AXES[i]].
# With the BGR input of the original pipeline, the network already predicts OpenGL normals (identity): checked on a
# synthetic Lambertian sphere (best of the 48 axis permutations / signs, ~3.5 deg mean error) and on real data
# (silhouette normals pointing outwards, see mrUniMSPS/tests/check_real_pose.py).
NATIVE_TO_OPENGL_AXES = (0, 1, 2)
NATIVE_TO_OPENGL_SIGNS = (1.0, 1.0, 1.0)

_LOGGER = logging.getLogger(__name__)


class Predictor:
    """Uni-MS-PS network ready for inference."""

    def __init__(self, model, device, logger):
        self.model = model
        self.device = device
        self.logger = logger


def loadModel(weightsPath, useGpu=True, logger=None):
    """Load the uncalibrated Uni-MS-PS network.

    The network weights stay on the CPU: with a GPU, the blocks are moved to the GPU one at a time during the
    inference (original Uni-MS-PS inference mode).

    Raises if the checkpoint cannot be read or does not provide every weight of the network.
    """
    from Transformer_multi_res_7 import Transformer_multi_res_7

    logger = logger or _LOGGER
    if not weightsPath or not os.path.isfile(weightsPath):
        raise FileNotFoundError("Uni-MS-PS weights not found: '{}'".format(weightsPath))
    checkpoint = torch.load(weightsPath, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid Uni-MS-PS checkpoint '{}': not a state dict".format(weightsPath))

    model = Transformer_multi_res_7(c_in=3, batch_size_encoder=6, batch_size_transformer=8000)
    keys = model.load_state_dict(checkpoint, strict=False)
    if keys.missing_keys:
        raise RuntimeError("Invalid Uni-MS-PS checkpoint '{}': {} missing weights (e.g. {})".format(
            weightsPath, len(keys.missing_keys), keys.missing_keys[0]))
    if keys.unexpected_keys:
        logger.debug("Unused checkpoint entries: {}".format(keys.unexpected_keys))
    model.eval()

    useCuda = bool(useGpu and torch.cuda.is_available())
    if useGpu and not useCuda:
        logger.warning("No GPU available: running Uni-MS-PS on the CPU (slow).")
    model.set_inference_mode(use_cuda_eval_mode=useCuda)
    # Pinned memory (and the GPU cache cleaning that goes with it) requires CUDA
    model.use_pinned_memory = useCuda
    return Predictor(model, "cuda" if useCuda else "cpu", logger)


def maskBoundingBox(mask, margin):
    """Inclusive bounding box of the mask, enlarged by margin pixels and clipped to the image.

    Returns:
        (rowStart, rowEnd, colStart, colEnd), end excluded (numpy slice bounds).
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("Empty mask")
    height, width = mask.shape
    return (max(0, int(rows[0]) - margin), min(height, int(rows[-1]) + 1 + margin),
            max(0, int(cols[0]) - margin), min(width, int(cols[-1]) + 1 + margin))


def canvasSize(height, width):
    """Side of the square network input for a crop: the smallest BASE_SIZE * 2^k >= max(height, width).

    Returns:
        (side, number of resolution stages k + 1)
    """
    nbStages = 1
    while BASE_SIZE * 2 ** (nbStages - 1) < max(height, width):
        nbStages += 1
    return BASE_SIZE * 2 ** (nbStages - 1), nbStages


def nativeToOpenGL(normal):
    """Convert normals from the native frame of the network to the OpenGL camera frame."""
    return np.stack([sign * normal[..., axis] for axis, sign in zip(NATIVE_TO_OPENGL_AXES, NATIVE_TO_OPENGL_SIGNS)],
                    axis=-1).astype(np.float32)


def predictNative(predictor, images, mask, cropMargin=0):
    """Normal map of one pose in the native frame of the network (see predict)."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("The mask must be a 2D array, got shape {}".format(mask.shape))
    if not images:
        raise ValueError("No image")
    if cropMargin < 0:
        raise ValueError("Negative crop margin: {}".format(cropMargin))
    height, width = mask.shape
    rowStart, rowEnd, colStart, colEnd = maskBoundingBox(mask, int(cropMargin))
    cropMask = mask[rowStart:rowEnd, colStart:colEnd]
    cropHeight, cropWidth = cropMask.shape
    size, nbStages = canvasSize(cropHeight, cropWidth)
    top, left = (size - cropHeight) // 2, (size - cropWidth) // 2
    rows, cols = slice(top, top + cropHeight), slice(left, left + cropWidth)

    # Network input: 1 x 3 (BGR) x nbImages x size x size, zero padding around the crop
    stack = np.zeros((3, len(images), size, size), np.float32)
    for index, image in enumerate(images):
        image = np.asarray(image, np.float32)
        if image.shape != (height, width, 3):
            raise ValueError("Image {} has shape {}, expected {}".format(index, image.shape, (height, width, 3)))
        crop = image[rowStart:rowEnd, colStart:colEnd, ::-1]  # RGB -> BGR, as the original OpenCV-based loader
        if not np.isfinite(crop).all():
            raise ValueError("Image {} has non-finite values in the processed area".format(index))
        # Original normalization: division by the mean intensity (mean of the channels) over the mask
        mean = float(crop.mean(axis=2)[cropMask].mean())
        if mean > 0:
            crop = crop / mean
        stack[:, index, rows, cols] = np.moveaxis(crop, 2, 0)
    canvasMask = np.zeros((size, size), bool)
    canvasMask[rows, cols] = cropMask

    predictor.logger.debug("Uni-MS-PS: crop rows [{}, {}), cols [{}, {}), network input {}x{} ({} stages)".format(
        rowStart, rowEnd, colStart, colEnd, size, size, nbStages))
    inputs = {"imgs": torch.from_numpy(stack).unsqueeze(0),
              "mask": torch.from_numpy(canvasMask).unsqueeze(0).unsqueeze(0)}
    with torch.no_grad():
        output = predictor.model.process(inputs, nbStages)["n"]
    output = output.detach().float().cpu()
    if tuple(output.shape) != (1, 3, size, size):
        raise RuntimeError("Unexpected Uni-MS-PS output shape {} (expected {})".format(
            tuple(output.shape), (1, 3, size, size)))
    output = output[0].permute(1, 2, 0).numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("Uni-MS-PS returned non-finite normals")

    # Depadding and uncrop
    normal = np.zeros((height, width, 3), np.float32)
    normal[rowStart:rowEnd, colStart:colEnd] = output[rows, cols]
    normal[~mask] = 0.0
    return normal


def predict(predictor, images, mask, cropMargin=0):
    """Normal map of one pose.

    Args:
        cropMargin: margin (pixels) around the inclusive bounding box of the mask (0: tight box, as the original
            Uni-MS-PS inference). The crop is padded with zeros to a square of side 32 * 2^k.
    """
    return {"normal": nativeToOpenGL(predictNative(predictor, images, mask, cropMargin=cropMargin))}
