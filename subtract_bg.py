"""
    Automatic Image Background Subtraction via benzin.io/remove.bg service.
"""

import os
import argparse
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
from typing import Tuple, Optional


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
    parser.add_argument("--threshold", type=int, default=127, help="threshold for alpha mask binarization")
    parser.add_argument("--input", type=str, required=True, help="input images directory path")
    parser.add_argument("--output", type=str, required=True, help="output masks directory path")
    parser.add_argument("--middle", type=str, required=False, help="optional directory path for raw masks from service")
    parser.add_argument("--ppdir", action="store_true", default=False,
                        help="add extra parrent+parrent directory to the output one")
    parser.add_argument("--jpg", action="store_true", help="optional forced recompression an input image as JPG")
    parser.add_argument("--not-resize", action="store_true", default=False,
                        help="suppress to forcibly scale the mask to the input image")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
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
                            input_image_path,
                            do_jpeg_recompress):
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
    do_jpeg_recompress : bool
        Whether to do forced recompression an input image as JPG.

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
    if do_jpeg_recompress:
        image = cv2.imread(input_image_path, flags=cv2.IMREAD_UNCHANGED)
        file = cv2.imencode(".jpg", image)[1].tobytes()
    else:
        file = open(input_image_path, "rb").read()
    response = requests.post(
        url=service_url,
        files={"image_file": file},
        data=data,
        headers={"X-Api-Key": token},
    )
    if response.status_code == requests.codes.ok:
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(buf=image, flags=cv2.IMREAD_UNCHANGED)
    else:
        raise RuntimeError("Error: status={}, text={}".format(response.status_code, response.text))
    return image


def create_output_file_path(src_file_path,
                            output_dir_path,
                            add_ppdir):
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


def py3round(number):
    """
    Unified rounding in all python versions.
    """
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(0.5 * number))
    else:
        return int(round(number))


def resize_image_with_max_size(image: np.ndarray,
                               max_size: int,
                               interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = image.shape[:2]
    scale = max_size / float(max(width, height))
    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        image = cv2.resize(image, dsize=(new_width, new_height), interpolation=interpolation)
    return image, (width, height)


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


if __name__ == "__main__":
    args = parse_args()
    service = args.service
    token = args.token
    url = args.url
    threshold = args.threshold
    input_image_dir_path = args.input
    output_mask_dir_path = args.output
    raw_mask_dir_path = args.middle
    add_ppdir = args.ppdir
    jpg = args.jpg
    preserve_size = not args.not_resize
    debug_mode = args.debug

    assert (service in ("benzinio", "removebg"))
    assert (token is not None) and (type(token) is str) and (len(token) > 0)
    assert (threshold is not None) and (type(threshold) is int) and (0 <= threshold <= 255)

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

    if raw_mask_dir_path is not None:
        raw_mask_dir_path = os.path.expanduser(raw_mask_dir_path)
        if not os.path.exists(raw_mask_dir_path):
            os.mkdir(raw_mask_dir_path)

    image_file_path_list = get_image_file_paths_with_subdirs(input_image_dir_path)

    for image_file_path in tqdm(image_file_path_list):
        image = cv2.imread(image_file_path, flags=cv2.IMREAD_UNCHANGED)

        mask_file_path = create_output_file_path(
            src_file_path=image_file_path,
            output_dir_path=output_mask_dir_path,
            add_ppdir=add_ppdir)

        if os.path.exists(mask_file_path):
            if debug_mode:
                mask = cv2.imread(mask_file_path, flags=cv2.IMREAD_UNCHANGED)
                image_with_mask = create_image_with_mask(image=image, mask=mask, max_size=None)
                debug_file_path = create_dubug_file_path(mask_file_path)
                cv2.imwrite(debug_file_path, image_with_mask)
            continue

        raw_mask = None
        raw_mask_file_path = None

        if raw_mask_dir_path is not None:
            raw_mask_file_path = create_output_file_path(
                src_file_path=image_file_path,
                output_dir_path=raw_mask_dir_path,
                add_ppdir=add_ppdir)
            if os.path.exists(raw_mask_file_path):
                raw_mask = cv2.imread(raw_mask_file_path, flags=cv2.IMREAD_UNCHANGED)

        if raw_mask is None:
            raw_mask = create_mask_via_service(
                service=service,
                token=token,
                url=url,
                input_image_path=image_file_path,
                do_jpeg_recompress=jpg)
            if preserve_size and (raw_mask.shape[:2] != image.shape[:2]):
                raw_mask = cv2.resize(raw_mask, dsize=image.shape[:2][::-1])
            if raw_mask_dir_path is not None:
                cv2.imwrite(raw_mask_file_path, raw_mask)

        mask = ((raw_mask[:, :, 3] > threshold).astype(np.uint8) * 255).astype(np.uint8)

        cv2.imwrite(mask_file_path, mask)
