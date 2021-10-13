"""
    Image auxiliary functions.
"""

__all__ = ['resize_image', 'resize_image_with_max_size', 'pad_image_to_size', 'pad_image_by_factor', 'crop_image',
           'normalize_image', 'create_image_with_mask']

import numpy as np
import cv2
from typing import Tuple, Optional
from .math import round_float_math, calc_pad_value, calc_pad_value_for_factor


def resize_image(image: np.ndarray,
                 image_size: Tuple[int, int],
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image.

    Parameters:
    ----------
    image : np.array
        Original image.
    image_size : tuple of 2 int
        Image size (height x width).
    interpolation : int
        OpenCV interpolation mode.

    Returns:
    -------
    np.array
        Resized image.
    """
    image_height, image_width = image.shape[:2]
    width, height = image_size
    if (height == image_height) and (width == image_width):
        return image
    else:
        return cv2.resize(image, dsize=image_size, interpolation=interpolation)


def resize_image_with_max_size(image: np.ndarray,
                               max_size: int,
                               interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image with maximal size.

    Parameters:
    ----------
    image : np.array
        Original image.
    max_size : int
        Maximal size value.
    interpolation : int
        OpenCV interpolation mode.

    Returns:
    -------
    np.array
        Resized image.
    tuple of 2 int
        Original image size.
    """
    height, width = image.shape[:2]
    scale = max_size / float(max(width, height))
    if scale != 1.0:
        new_height, new_width = tuple(round_float_math(dim * scale) for dim in (height, width))
        image = cv2.resize(image, dsize=(new_width, new_height), interpolation=interpolation)
    return image, (width, height)


def pad_image_to_size(image: np.array,
                      dst_image_size: Tuple[int, int],
                      border_type: int = cv2.BORDER_REFLECT_101) -> Tuple[np.array, Tuple[int, int, int, int]]:
    """
    Pad the image.

    Parameters:
    ----------
    image : np.array
        Padding image.
    dst_image_size : tuple of 2 int
        Desired image size (height x width).
    border_type : int, default cv2.BORDER_REFLECT_101
        OpenCV border type.

    Returns:
    -------
    padded_image : np.array
        Padded image.
    tuple of 4 int
        Padding parameters.
    """
    dst_width, dst_height = dst_image_size
    height, width = image.shape[:2]
    y_left_pad, y_right_pad = calc_pad_value(height, dst_height)
    x_left_pad, x_right_pad = calc_pad_value(width, dst_width)
    padded_image = cv2.copyMakeBorder(image, y_left_pad, y_right_pad, x_left_pad, x_right_pad, border_type)
    return padded_image, (x_left_pad, y_left_pad, x_right_pad, y_right_pad)


def pad_image_by_factor(image: np.array,
                        factor: int = 32,
                        border_type: int = cv2.BORDER_REFLECT_101) -> Tuple[np.array, Tuple[int, int, int, int]]:
    """
    Pad the image on the sides, so that it will be divisible by factor.

    Parameters:
    ----------
    image : np.array
        Padding image.
    factor : int, default 32
        Alignment value.
    border_type : int, default cv2.BORDER_REFLECT_101
        OpenCV border type.

    Returns:
    -------
    padded_image : np.array
        Padded image.
    tuple of 4 int
        Padding parameters.
    """
    height, width = image.shape[:2]
    y_left_pad, y_right_pad = calc_pad_value_for_factor(height, factor)
    x_left_pad, x_right_pad = calc_pad_value_for_factor(width, factor)
    padded_image = cv2.copyMakeBorder(image, y_left_pad, y_right_pad, x_left_pad, x_right_pad, border_type)
    return padded_image, (x_left_pad, y_left_pad, x_right_pad, y_right_pad)


def crop_image(image: np.array,
               crop_params: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image patch from the center so that sides are equal to pads.

    Parameters:
    ----------
    image : np.array
        Cropping image.
    pad_params : tuple of 4 int
        Cropping parameters.

    Returns:
    -------
    np.array
        Cropped image.
    """
    height, width = image.shape[:2]
    x_left_pad, y_left_pad, x_right_pad, y_right_pad = crop_params
    return image[y_left_pad:(height - y_right_pad), x_left_pad:(width - x_right_pad)]


def normalize_image(image: np.array,
                    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                    max_pixel_value: float = 255.0) -> np.ndarray:
    """
    Normalize image (with default parameters for ImageNet normalization).

    Parameters:
    ----------
    image : np.array
        Cropping image.
    mean : tuple of 3 float, default (0.485, 0.456, 0.406)
        Mean value.
    std : tuple of 3 float, default (0.229, 0.224, 0.225)
        STD value.
    max_pixel_value : float, default 255.0
        Maximal value for source pixels.

    Returns:
    -------
    np.array
        Normalized image.
    """
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value
    denominator = np.reciprocal(std, dtype=np.float32)

    image = image.astype(np.float32)
    image -= mean
    image *= denominator
    return image


def create_image_with_mask(image: np.ndarray,
                           mask: np.ndarray,
                           max_size: Optional[int] = None) -> np.ndarray:
    """
    Create an image with mask for debug purpose.

    Parameters:
    ----------
    image : np.array
        Original RGB image.
    mask : np.array
        Mask image.
    max_size : int or None
        Maximal dimension for resulted image.

    Returns:
    -------
    np.array
        Destination image with mask.
    """
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, dsize=image.shape[:2][::-1])
        mask = (mask > 127).astype(np.uint8) * 255
        assert (mask.dtype == np.uint8)
    image_with_mask = cv2.addWeighted(
        src1=image,
        alpha=0.5,
        src2=(cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2RGB) * (0, 1, 0)).astype(np.uint8),
        beta=0.5,
        gamma=0.0)
    image_with_mask = cv2.addWeighted(
        src1=image_with_mask,
        alpha=0.75,
        src2=(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (1, 0, 0)).astype(np.uint8),
        beta=0.25,
        gamma=0.0)
    if max_size is not None:
        image_with_mask, _ = resize_image_with_max_size(image_with_mask, max_size=max_size)
    return image_with_mask
