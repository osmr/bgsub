"""
    Automatic Image Background Subtraction via benzin.io/remove.bg service.
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
    parser.add_argument("--service", type=str, default="benzinio",
                        help="service name. options are `benzinio` or `removebg`")
    parser.add_argument("--token", type=str, required=True, help="service API token")
    parser.add_argument("--url", type=str, help="optional custom URL for service")
    parser.add_argument("--threshold", type=int, default=127, help="threshol for alpha mask binarization")
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    parser.add_argument("--middle", type=str, help="optional directory path for raw masks from service")
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


def create_mask_via_service(service,
                            token,
                            url,
                            input_image_path):
    """
    Process image via benzin.io/remove.bg service.

    Parameters:
    ----------
    service : str
        Service name.
    token : str
        RemoteBG API token.
    url : str
        Optional custom URL for service.
    input_image_path : str
        Path to input image file.

    Returns:
    -------
    image : np.array
        Output mask.
    """
    default_url_dict = {
        "benzinio": "https://api.benzin.io/v1/removeBackground",
        "removebg": "https://api.remove.bg/v1.0/removebg"}
    service_url = url if url is not None else default_url_dict[service]
    if service == "benzinio":
        data = {"size": "full", "output_format": "image"}
    elif service == "removebg":
        data = {"size": "auto", "format": "auto"}
    else:
        raise NotImplemented("Wrong service name: {}".format(service))
    with open(input_image_path, "rb") as f:
        response = requests.post(
            url=service_url,
            files={"image_file": f},
            data=data,
            headers={"X-Api-Key": token},
        )
    if response.status_code == requests.codes.ok:
        # assert (output_mask_path is not None)
        # with open(output_mask_path, "wb") as out:
        #     out.write(response.content)
        # image = cv2.imread(dst_file_path)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(buf=image, flags=cv2.IMREAD_UNCHANGED)
    else:
        raise RuntimeError("Error: status={}, text={}".format(response.status_code, response.text))
    return image


if __name__ == "__main__":
    args = parse_args()
    service = args.service
    token = args.token
    url = args.url
    threshold = args.threshold
    input_image_dir_path = args.input
    output_mask_dir_path = args.output
    middle_mask_dir_path = args.middle

    assert (service in ("benzinio", "removebg"))
    assert (token is not None) and (type(token) is str) and (len(token) > 0)
    assert (threshold is not None) and (type(threshold) is int) and (0 <= threshold <= 255)

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

    middle_mask_dir_path = os.path.expanduser(middle_mask_dir_path)
    if not os.path.exists(middle_mask_dir_path):
        os.mkdir(middle_mask_dir_path)

    image_file_path_list = get_image_file_paths_with_subdirs(input_image_dir_path)

    for image_file_path in tqdm(image_file_path_list):
        src_file_stem, _ = os.path.splitext(image_file_path)
        dst_file_stem_path = os.path.join(output_mask_dir_path, os.path.basename(src_file_stem))
        dst_file_path = "{}.png".format(dst_file_stem_path)

        image = cv2.imread(image_file_path, flags=cv2.IMREAD_UNCHANGED)

        if os.path.exists(dst_file_path):
            mask = cv2.imread(dst_file_path, flags=cv2.IMREAD_UNCHANGED)
            # cv2.imshow("image", image)
            # cv2.imshow("mask", mask)
            # cv2.waitKey()
            continue

        mask_raw = None
        mask_raw_file_path = None
        if middle_mask_dir_path is not None:
            mask_raw_file_path = "{}.png".format(os.path.join(middle_mask_dir_path, os.path.basename(src_file_stem)))
            if os.path.exists(mask_raw_file_path):
                mask_raw = cv2.imread(mask_raw_file_path, flags=cv2.IMREAD_UNCHANGED)

        if mask_raw is None:
            mask_raw = create_mask_via_service(
                service=service,
                token=token,
                url=url,
                input_image_path=image_file_path)
            if middle_mask_dir_path is not None:
                cv2.imwrite(mask_raw_file_path, mask_raw)

        mask = ((mask_raw[:, :, 3] >= threshold).astype(np.uint8) * 255).astype(np.uint8)

        # cv2.imshow("image", image)
        # cv2.imshow("mask_raw", mask_raw)
        # cv2.imshow("mask", mask)
        # cv2.waitKey()

        cv2.imwrite(dst_file_path, mask)
