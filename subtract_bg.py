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
    parser.add_argument("--middle", type=str, required=True, help="optional directory path for raw masks from service")
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

    if debug_mode:
        cv2.namedWindow("debug", flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("debug", 2000, 600)

    assert (service in ("benzinio", "removebg"))
    assert (token is not None) and (type(token) is str) and (len(token) > 0)
    assert (threshold is not None) and (type(threshold) is int) and (0 <= threshold <= 255)

    input_image_dir_path = os.path.expanduser(input_image_dir_path)
    if not os.path.exists(input_image_dir_path):
        raise Exception("Input image directory doesn't exist: {}".format(input_image_dir_path))

    output_mask_dir_path = os.path.expanduser(output_mask_dir_path)
    if not os.path.exists(output_mask_dir_path):
        os.mkdir(output_mask_dir_path)

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
                raw_mask = cv2.resize(raw_mask, dsize=image.shape[:2])
            if raw_mask_dir_path is not None:
                cv2.imwrite(raw_mask_file_path, raw_mask)

        mask = ((raw_mask[:, :, 3] >= threshold).astype(np.uint8) * 255).astype(np.uint8)

        if debug_mode:
            # NOTE(i.rodin): Assuming we have all masks similar sizes
            mask_to_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_as_alpha = mask_to_show.copy()
            mask_as_alpha[mask_as_alpha > 0] = 1
            stack_image = np.hstack([image, mask_to_show, cv2.multiply(image, mask_as_alpha)])
            cv2.imshow("debug", stack_image)
            cv2.waitKey()

        cv2.imwrite(mask_file_path, mask)
