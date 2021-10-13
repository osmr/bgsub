"""
    Pytorch auxiliary functions.
"""

__all__ = ['rename_layer_names_in_state_dict', 'create_tensor_from_rgb_image']

import re
import torch
import numpy as np


def rename_layer_names_in_state_dict(state_dict,
                                     layer_name_dict: dict):
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


def create_tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    """
    Create a Pytorch tensor from a RGB image.

    Parameters:
    ----------
    image : np.array
        RGB image.

    Returns:
    -------
    torch.Tensor
        Destination tensor.
    """
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)
