"""
    Mathematical auxiliary functions.
"""

__all__ = ['round_float_math', 'calc_pad_value', 'calc_pad_value_for_factor']

from typing import Tuple


def round_float_math(number: float) -> int:
    """
    Unified rounding in all python versions.

    Parameters:
    ----------
    number : float
        Input number.

    Returns:
    -------
    int
        Output rounded number.
    """
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(0.5 * number))
    else:
        return int(round(number))


def calc_pad_value(src_value: int,
                   dst_value: int) -> Tuple[int, int]:
    """
    Calculate a padding values for a pair of source and destination numbers.

    Parameters:
    ----------
    src_value : int
        Source number.
    dst_value : int
        Destination number.

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


def calc_pad_value_for_factor(value: int,
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
