
from __future__ import print_function

import numpy as np
import cv2
from imaugtools import center_crop, crop_around_center
from PIL import Image, ImageFilter


def image_zoom(image, param=1.5):
    """ param: 1-2 """
    res = cv2.resize(image, None, fx=param, fy=param, interpolation=cv2.INTER_LINEAR)
    # res = crop_around_center(res, (image.shape))
    if param >= 1:
        res = crop_around_center(res, (image.shape))
    else:
        res = center_crop(res, (image.shape))
    return res


def image_blur(image, params=2):
    # blur = cv2.blur(image, (params+1, params+1))
    image_data = image.squeeze().astype("uint8")
    image = Image.fromarray(image_data)
    gaussImage = image.filter(ImageFilter.GaussianBlur(params))
    image_blur = np.asarray(gaussImage)
    # image_blur = skimage.filters.gaussian(
    #     image.squeeze(), sigma=(1.0, 1.0), truncate=2, multichannel=True)
    return image_blur


def image_brightness(image, param=128):
    image = np.int16(image)
    image = image + param
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image


def image_contrast(image, param=1.5):
    """
    param: 0-2
    """
    image = np.int16(image)
    image = (image-127) * param + 127
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image


def image_translation_cropped(img, params):
    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, ch = img.shape

    M = np.float32([[1, 0, params], [0, 1, params]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_shear_cropped(img, params):
    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [factor, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst