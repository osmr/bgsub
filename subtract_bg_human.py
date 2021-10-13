"""
    Automatic Image Background Subtraction via human segmentation neural net.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
from segmentation_models_pytorch import Unet
from torch.utils import model_zoo
import segmentation_refinement as refine
import torch
from utils.path import get_image_file_paths_with_subdirs, create_output_file_path, create_dubug_file_path
from utils.image import resize_image, resize_image_with_max_size, pad_image_to_size, pad_image_by_factor, crop_image,\
    normalize_image, create_image_with_mask
from utils.pytorch import rename_layer_names_in_state_dict, create_tensor_from_rgb_image


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
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    parser.add_argument("--ppdir", action="store_true", default=False,
                        help="add extra parrent+parrent directory to the output one")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    args = parser.parse_args()
    return args


def create_human_segm_net():
    """
    Create human segmentation Pytorch network.

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

    net = create_human_segm_net()
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

        padded_image, factor_pad_params = pad_image_by_factor(
            image=rgb_image,
            factor=32,
            border_type=cv2.BORDER_CONSTANT)
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
