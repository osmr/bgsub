"""
    Automatic Image Background Subtraction via RemoteBG service.
"""

import os
import argparse
import requests
from tqdm import tqdm
import cv2
import numpy as np


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
    parser.add_argument("--token", type=str, required=True, help="RemoveBG API token")
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    args = parser.parse_args()
    return args


def get_image_file_paths_with_subdirs(dir_path):
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


def get_mask_via_remotebg(token,
                          input_image_path,
                          output_mask_path):
    """
    Process image via RemoteBG service.

    Parameters:
    ----------
    token : str
        RemoteBG API token.
    input_image_path : str
        Path to input image file.
    output_mask_path : str
        Path to output mask file.

    Returns:
    -------
    image : np.array
        Output mask.
    """
    assert (output_mask_path is not None)
    response = requests.post(
        "https://api.remove.bg/v1.0/removebg",
        files={'image_file': open(input_image_path, "rb")},
        data={"size": "auto"},
        headers={"X-Api-Key": token},
    )
    if response.status_code == requests.codes.ok:
        # with open(output_mask_path, "wb") as out:
        #     out.write(response.content)
        # image = cv2.imread(dst_file_path)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        raise RuntimeError("Error:", response.status_code, response.text)
    return image


if __name__ == "__main__":
    args = parse_args()
    token = args.token
    input_image_dir_path = args.input
    output_mask_dir_path = args.output

    assert (token is not None) and (type(token) is str) and (len(token) > 0)

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

    image_file_path_list = get_image_file_paths_with_subdirs(input_image_dir_path)

    for image_file_path in tqdm(image_file_path_list):
        src_file_stem, _ = os.path.splitext(image_file_path)
        dst_file_stem_path = os.path.join(output_mask_dir_path, os.path.basename(src_file_stem))
        dst_file_path = "{}.png".format(dst_file_stem_path)

        mask_rgb = get_mask_via_remotebg(
            token=token,
            input_image_path=image_file_path,
            output_mask_path=dst_file_path)
        mask = ((mask_rgb[:, :, 0] > 0).astype(np.uint8) * 255).astype(np.uint8)
        cv2.imwrite(dst_file_path, mask)

        if not os.path.exists(dst_file_path):
            mask_rgb = get_mask_via_remotebg(
                token=token,
                input_image_path=image_file_path,
                output_mask_path=dst_file_path)
            mask = ((mask_rgb[:, :, 0] > 0).astype(np.uint8) * 255).astype(np.uint8)
            cv2.imwrite(dst_file_path, mask)
        else:
            image = cv2.imread(dst_file_path)
            mask = ((image[:, :, 0] > 0).astype(np.uint8) * 255).astype(np.uint8)
            cv2.imshow("image", image)
            cv2.imshow("mask", mask)
            cv2.waitKey()
