"""
    Path auxiliary functions.
"""

__all__ = ['get_image_file_paths_with_subdirs', 'create_output_file_path', 'create_dubug_file_path']

import os
from pathlib import Path
from typing import List


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


def create_dubug_file_path(file_path: str,
                           dst_file_ext: str = ".jpg") -> str:
    """
    Create a file path for debug image.

    Parameters:
    ----------
    file_path : str
        Original image file path.
    dst_file_ext : str, default '.jpg"'
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
