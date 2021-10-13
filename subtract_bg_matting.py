"""
    Automatic Image Background Subtraction via background matting network.
"""

import os
import argparse
from tqdm import tqdm
import cv2
from utils.path import get_image_file_paths_with_subdirs, create_output_file_path, create_dubug_file_path
from utils.image import create_image_with_mask
from matting_bg_subtractor import MatthingBgSubtractor


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
    parser.add_argument("--threshold", type=int, default=127, help="threshold for matting mask binarization")
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--bg", type=str, required=True, help="background image file path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    parser.add_argument("--ppdir", action="store_true", default=False,
                        help="add extra parrent+parrent directory to the output one")
    parser.add_argument("--use-cuda", action="store_true", help="use CUDA")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    threshold = args.threshold
    input_image_dir_path = args.input
    bg_image_file_path = args.bg
    output_mask_dir_path = args.output
    add_ppdir = args.ppdir
    debug_mode = args.debug
    use_cuda = args.use_cuda

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    bg_image_file_path = os.path.expanduser(bg_image_file_path)
    if not os.path.exists(bg_image_file_path):
        raise Exception("Background image file doesn't exist: {}".format(bg_image_file_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

    bgsub_net = MatthingBgSubtractor(threshold=threshold, use_cuda=use_cuda)

    image_file_path_list = get_image_file_paths_with_subdirs(input_image_dir_path)

    bg_bgr_image = cv2.imread(bg_image_file_path, flags=cv2.IMREAD_UNCHANGED)
    bg_rgb_image = cv2.cvtColor(bg_bgr_image, code=cv2.COLOR_BGR2RGB)

    for image_file_path in tqdm(image_file_path_list):
        mask_file_path = create_output_file_path(
            src_file_path=image_file_path,
            output_dir_path=output_mask_dir_path,
            add_ppdir=add_ppdir)

        if os.path.exists(mask_file_path):
            continue

        bgr_image = cv2.imread(image_file_path, flags=cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.cvtColor(bgr_image, code=cv2.COLOR_BGR2RGB)

        mask = bgsub_net(rgb_image, bg_rgb_image)
        cv2.imwrite(mask_file_path, mask)

        if debug_mode:
            image_with_mask = create_image_with_mask(image=bgr_image, mask=mask, max_size=None)
            debug_file_path = create_dubug_file_path(mask_file_path)
            cv2.imwrite(debug_file_path, image_with_mask)
