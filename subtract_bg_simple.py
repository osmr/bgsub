"""
    Image Background Subtraction via simple image difference.
"""

import os
import argparse
import numpy as np
import cv2
from utils.path import create_dubug_file_path
from utils.image import create_image_with_mask


def parse_args() -> argparse.Namespace:
    """
    Parse python script parameters.

    Returns:
    -------
    argparse.Namespace
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Automatic Image Background Subtraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="input image file path")
    parser.add_argument("--bg", type=str, required=True, help="input background image file path")
    parser.add_argument("--output", type=str, required=True, help="output mask file path")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    args = parser.parse_args()
    return args


def convert_srgb_to_linear(srgb_image: np.ndarray) -> np.ndarray:
    linear_image = np.float32(srgb_image) / 255.0
    less_thr = (linear_image <= 0.04045)
    linear_image[less_thr] = linear_image[less_thr] / 12.92
    linear_image[~less_thr] = np.power((linear_image[~less_thr] + 0.055) / 1.055, 2.4)
    return linear_image


def convert_linear_to_srgb(linear_image: np.ndarray) -> np.ndarray:
    srgb_image = linear_image.copy()
    less_thr = (linear_image <= 0.0031308)
    srgb_image[less_thr] = linear_image[less_thr] * 12.92
    srgb_image[~less_thr] = 1.055 * np.power(linear_image[~less_thr], 1.0 / 2.4) - 0.055
    return srgb_image * 255.0


if __name__ == "__main__":
    args = parse_args()
    input_image_file_path = args.input
    bg_image_file_path = args.bg
    output_mask_file_path = args.output
    debug_mode = args.debug

    input_image_file_path = os.path.expanduser(input_image_file_path)
    if not os.path.exists(input_image_file_path):
        raise Exception("Input image file doesn't exist: {}".format(input_image_file_path))

    bg_image_file_path = os.path.expanduser(bg_image_file_path)
    if not os.path.exists(bg_image_file_path):
        raise Exception("Input background image file doesn't exist: {}".format(bg_image_file_path))

    output_mask_file_path = os.path.expanduser(output_mask_file_path)

    bgr_input_image = cv2.imread(input_image_file_path, flags=cv2.IMREAD_COLOR)
    bgr_bg_image = cv2.imread(bg_image_file_path, flags=cv2.IMREAD_COLOR)

    use_linear = False
    use_lab = True
    if use_linear:
        srgb_input_image = cv2.cvtColor(bgr_input_image, code=cv2.COLOR_BGR2RGB)
        srgb_bg_image = cv2.cvtColor(bgr_bg_image, code=cv2.COLOR_BGR2RGB)

        lrgb_input_image = convert_srgb_to_linear(srgb_input_image)
        lrgb_bg_image = convert_srgb_to_linear(srgb_bg_image)

        diff = np.maximum(lrgb_input_image, lrgb_bg_image) - np.minimum(lrgb_input_image, lrgb_bg_image)
        diff = convert_linear_to_srgb(diff)

        diff = cv2.cvtColor(diff, code=cv2.COLOR_RGB2BGR)
    elif use_lab:
        lab_input_image = cv2.cvtColor(bgr_input_image, code=cv2.COLOR_BGR2Lab)
        lab_bg_image = cv2.cvtColor(bgr_bg_image, code=cv2.COLOR_BGR2Lab)

        diff = np.maximum(lab_input_image, lab_bg_image) - np.minimum(lab_input_image, lab_bg_image)

        diff = cv2.cvtColor(diff, code=cv2.COLOR_Lab2BGR)
    else:
        diff = np.maximum(bgr_input_image, bgr_bg_image) - np.minimum(bgr_input_image, bgr_bg_image)

    cv2.imwrite(output_mask_file_path, diff)

    # if debug_mode:
    #     image_with_mask = create_image_with_mask(image=bgr_image, mask=mask, max_size=None)
    #     debug_file_path = create_dubug_file_path(output_mask_file_path)
    #     cv2.imwrite(debug_file_path, image_with_mask)
