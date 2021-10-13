"""
    Automatic Image Background Subtraction via human segmentation neural net.
"""

import os
import re
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
from segmentation_models_pytorch import Unet
from torch.utils import model_zoo
import segmentation_refinement as refine
from typing import List, Tuple, Optional
import torch


def parse_args():
    """
    Parse python script parameters.

    Returns:
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Automatic Image Background Subtraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    parser.add_argument("--ppdir", action="store_true", default=False,
                        help="add extra parrent+parrent directory to the output one")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    args = parser.parse_args()
    return args


def get_image_file_paths_with_subdirs(dir_path: str) -> List[str]:
    """
    Get all image file paths.

    Parameters:
    ----------
    dir_path : str
        Path to working directory.

    Returns:
    -------
    list of str
        Paths to image files.
    """
    image_file_paths = []
    for subdir, dirs, files in os.walk(dir_path):
        for file_name in files:
            _, file_ext = os.path.splitext(file_name)
            if file_ext.lower() == ".jpg":
                image_file_path = os.path.join(subdir, file_name)
                image_file_paths.append(image_file_path)
    image_file_paths = sorted(image_file_paths)
    return image_file_paths


def rename_layer_names_in_state_dict(state_dict,
                                     layer_name_dict):
    """
    Rename layer names in a Pytorch model state dict.

    Parameters:
    ----------
    state_dict : OrderedDict
        Source Pytorch model state dict.
    layer_name_dict : dict
        Renamed layers.

    Returns:
    -------
    dst_state_dict : OrderedDict
        Updated state dict.
    """
    dst_state_dict = {}
    for key, value in state_dict.items():
        for key_l, value_l in layer_name_dict.items():
            key = re.sub(key_l, value_l, key)
        dst_state_dict[key] = value
    return dst_state_dict


def create_net():
    """
    Create human segmentation Pytorch net.

    Returns:
    -------
    net : nn.Module
        Target net with loaded weights.
    """
    model_weights_url = "https://github.com/ternaus/people_segmentation/releases/download/0.0.1/2020-09-23a.zip"
    net = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
    state_dict = model_zoo.load_url(url=model_weights_url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layer_names_in_state_dict(state_dict=state_dict, layer_name_dict={"model.": ""})
    net.load_state_dict(state_dict)
    net.eval()
    return net


def py3round(number):
    """
    Unified rounding in all python versions.
    """
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(0.5 * number))
    else:
        return int(round(number))


def resize_image(image: np.ndarray,
                 image_size: Tuple[int, int],
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    width, height = image_size
    if (height == image_height) and (width == image_width):
        return image
    else:
        return cv2.resize(image, dsize=image_size, interpolation=interpolation)


def resize_image_with_max_size(image: np.ndarray,
                               max_size: int,
                               interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = image.shape[:2]
    scale = max_size / float(max(width, height))
    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        image = cv2.resize(image, dsize=(new_width, new_height), interpolation=interpolation)
    return image, (width, height)


def calc_pad_value(src_value: int,
                   dst_value: int) -> Tuple[int, int]:
    """
    Pad a value, so that it will be divisible by factor.

    Parameters:
    ----------
    value : int
        Padded value.
    factor : int
        Alignment value.

    Returns:
    -------
    tuple of 2 int
        Left and right paddings.
    """
    assert (dst_value >= src_value)
    if src_value == dst_value:
        pad_left = 0
        pad_right = 0
    else:
        pad_value = dst_value - src_value
        pad_left = pad_value // 2
        pad_right = pad_value - pad_left
    return pad_left, pad_right


def pad_image_to_size(image: np.array,
                      dst_image_size: Tuple[int, int],
                      border_type: int = cv2.BORDER_REFLECT_101) -> Tuple[np.array, Tuple[int, int, int, int]]:
    """
    Pad the image.

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
    dst_width, dst_height = dst_image_size
    height, width = image.shape[:2]
    y_left_pad, y_right_pad = calc_pad_value(height, dst_height)
    x_left_pad, x_right_pad = calc_pad_value(width, dst_width)
    padded_image = cv2.copyMakeBorder(image, y_left_pad, y_right_pad, x_left_pad, x_right_pad, border_type)
    return padded_image, (x_left_pad, y_left_pad, x_right_pad, y_right_pad)


def pad_value(value: int,
              factor: int) -> Tuple[int, int]:
    """
    Pad a value, so that it will be divisible by factor.

    Parameters:
    ----------
    value : int
        Padded value.
    factor : int
        Alignment value.

    Returns:
    -------
    tuple of 2 int
        Left and right paddings.
    """
    if value % factor == 0:
        pad_left = 0
        pad_right = 0
    else:
        pad_value = factor - value % factor
        pad_left = pad_value // 2
        pad_right = pad_value - pad_left
    return pad_left, pad_right


def pad_image(image: np.array,
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
    y_left_pad, y_right_pad = pad_value(height, factor)
    x_left_pad, x_right_pad = pad_value(width, factor)
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


def create_tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    """
    Create a Pytorch tensor from a RGB image.

    Parameters:
    ----------
    image : np.array
        Cropping image.

    Returns:
    -------
    torch.Tensor
        Destination tensor.
    """
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


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


def create_dubug_file_path(file_path: str,
                           dst_file_ext: Optional[str] = ".jpg") -> str:
    """
    Create a file path for debug image.

    Parameters:
    ----------
    file_path : str
        Original image file path.
    dst_file_ext : str
        Destination file extension.

    Returns:
    -------
    str
        Destination image file path.
    """
    src_file_stem, src_file_ext = os.path.splitext(file_path)
    dst_file_ext = dst_file_ext if dst_file_ext is not None else src_file_ext
    dst_file_path = "{}_debug{}".format(src_file_stem, dst_file_ext)
    return dst_file_path


def create_output_file_path(src_file_path: str,
                            output_dir_path: str,
                            add_ppdir: bool) -> str:
    """
    Create path to output file (mask).

    Parameters:
    ----------
    dst_file_path : str
        Path to an input file (image).
    output_dir_path : str
        Path to output base directory.
    add_ppdir : bool
        Whether to add extra parrent+parrent directory to the output one.

    Returns:
    -------
    dst_file_path : str
        Path to output file (mask).
    """
    src_file_stem, _ = os.path.splitext(src_file_path)
    if add_ppdir:
        pp_dir_name = Path(src_file_stem).parent.parent.name
        real_output_dir_path = os.path.join(output_dir_path, pp_dir_name)
        if not os.path.exists(real_output_dir_path):
            os.mkdir(real_output_dir_path)
    else:
        real_output_dir_path = output_dir_path
    dst_file_stem_path = os.path.join(real_output_dir_path, os.path.basename(src_file_stem))
    dst_file_path = "{}.png".format(dst_file_stem_path)
    return dst_file_path


if __name__ == "__main__":
    args = parse_args()
    input_image_dir_path = args.input
    output_mask_dir_path = args.output
    add_ppdir = args.ppdir
    debug_mode = args.debug

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

    net = create_net()
    # refiner = refine.Refiner(device="cpu")
    refiner = refine.Refiner(device='cuda:0')

    image_file_path_list = get_image_file_paths_with_subdirs(input_image_dir_path)

    for image_file_path in tqdm(image_file_path_list):
        mask_file_path = create_output_file_path(
            src_file_path=image_file_path,
            output_dir_path=output_mask_dir_path,
            add_ppdir=add_ppdir)

        if os.path.exists(mask_file_path):
            continue

        bgr_image = cv2.imread(image_file_path, flags=cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.cvtColor(bgr_image, code=cv2.COLOR_BGR2RGB)

        base_size = 800
        rgb_image, rgb_image_size = resize_image_with_max_size(rgb_image, max_size=base_size)
        rgb_image, rgb_image_pad_params = pad_image_to_size(
            image=rgb_image,
            dst_image_size=(base_size, base_size),
            border_type=cv2.BORDER_CONSTANT)

        padded_image, factor_pad_params = pad_image(image=rgb_image, factor=32, border_type=cv2.BORDER_CONSTANT)
        x = normalize_image(image=padded_image)
        x = create_tensor_from_rgb_image(x)
        x = torch.unsqueeze(x, dim=0)

        with torch.no_grad():
            y = net(x)[0][0]

        mask = (y > 0).cpu().detach().numpy().astype(np.uint8)
        mask = mask * 255

        mask = refiner.refine(rgb_image, mask, fast=True, L=1200)

        mask = crop_image(mask, factor_pad_params)

        mask = crop_image(mask, rgb_image_pad_params)
        mask = resize_image(mask, image_size=rgb_image_size)

        mask = refiner.refine(bgr_image, mask, fast=True, L=1800)
        mask = (mask > 127).astype(np.uint8) * 255
        assert (mask.dtype == np.uint8)

        cv2.imwrite(mask_file_path, mask)

        if debug_mode:
            image_with_mask = create_image_with_mask(image=bgr_image, mask=mask, max_size=None)
            debug_file_path = create_dubug_file_path(mask_file_path)
            cv2.imwrite(debug_file_path, image_with_mask)

            # cv2.imshow("image", bgr_image)
            # cv2.imshow("mask", mask * 255)
            # cv2.imshow("image_with_mask", resize_image_with_max_size(image_with_mask, max_size=1280)[0])
            # cv2.imshow("image_with_mask2", resize_image_with_max_size(create_image_with_mask(image=bgr_image, mask=mask2), max_size=1280)[0])  # noqa
            # cv2.waitKey()
