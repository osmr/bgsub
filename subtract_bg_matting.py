"""
    Automatic Image Background Subtraction via human segmentation neural net.
"""

import os
import argparse
from tqdm import tqdm
import cv2
from utils.path import get_image_file_paths_with_subdirs, create_output_file_path, create_dubug_file_path
from utils.image import create_image_with_mask
from image_human_segmenter import ImageHumanSegmenter
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
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    parser.add_argument("--ppdir", action="store_true", default=False,
                        help="add extra parrent+parrent directory to the output one")
    parser.add_argument("--use-cuda", action="store_true", help="use CUDA")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    input_image_dir_path = args.input
    output_mask_dir_path = args.output
    add_ppdir = args.ppdir
    debug_mode = args.debug
    use_cuda = args.use_cuda

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

    # segmenter = ImageHumanSegmenter(use_cuda)
    segmenter = MatthingBgSubtractor(use_cuda)

    image_file_path_list = get_image_file_paths_with_subdirs(input_image_dir_path)

    for image_file_path in tqdm(image_file_path_list):
        mask_file_path = create_output_file_path(
            src_file_path=image_file_path,
            output_dir_path=output_mask_dir_path,
            add_ppdir=add_ppdir)

        if os.path.exists(mask_file_path):
            continue

        bgr_image = cv2.imread(image_file_path, flags=cv2.IMREAD_UNCHANGED)
        mask = segmenter(bgr_image)
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
