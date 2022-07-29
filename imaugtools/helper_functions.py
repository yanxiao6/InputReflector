import numpy as np
from typing import Union


def _get_dtype(image):
    dtype = type(image)
    if hasattr(image, 'dtype'):
        if type(image.dtype) == np.dtype:
            dtype = image.dtype
        else:
            dtype = image.dtype.as_numpy_dtype
    return dtype


def _convert_tensor_to_numpy_if_possible(image):
    if(hasattr(image, 'dtype') and
            type(image.dtype) != np.dtype and
            hasattr(image, 'numpy')):
        return image.numpy()
    return image


def _get_largest_rotated_rectangle(height, width, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    Parameters:
        height: Input height in pixels
        width: Input width in pixels
        angle: radians
    Returns:
        (float, float): Maximal height and width that can be cropped around the center
    """
    if width <= 0 or height <= 0:
        return 0., 0.

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a

    return hr, wr
