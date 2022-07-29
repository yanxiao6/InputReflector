import numpy as np
from imaugtools.helper_functions import _get_dtype, _convert_tensor_to_numpy_if_possible


def translate_image(image, tx, ty, tx_max=None, ty_max=None, crop=True):

    """
    Given a NumPy / OpenCV 2 image, center crops it to the given size.
    Parameters:
        image: Input Image: numpy array, eagertensor or cv2 mat
        tx: can vary from 0 to tx_max
        ty: can vary from 0 to ty_max
        tx_max: max translation window size on left and right
        ty_max: max translation window size on top and bottom
        crop: can be false when tx and ty are not specified
    Returns:
        image: Output Image, same size if crop is True, else cropped
    """

    # For tensor processing
    image = _convert_tensor_to_numpy_if_possible(image)

    if not crop and ty_max is not None and tx_max is not None:
        raise ValueError("Can't have crop=False when tx_max or ty_max is specified")
    if tx_max is None: tx_max = abs(tx)
    if ty_max is None: ty_max = abs(ty)

    # predict the size from tx_max and ty_max and resize
    size = (int(image.shape[0] / (1 + 2 * ty_max)), int(image.shape[1] / (1 + 2 * tx_max)))

    # get offsets for centered image
    top_offset = int((image.shape[0] - size[0]) // 2)
    left_offset = int((image.shape[1] - size[1]) // 2)

    # calculate new offsets based on tx and ty
    top_offset = top_offset + int(size[0] * ty)
    left_offset = left_offset + int(size[1] * tx)

    # if(top_offset < 0 or
    #         left_offset < 0 or
    #         top_offset + size[0] > image.shape[0] or
    #         left_offset + size[1] > image.shape[1]):
    #     reason = "None"
    #     if top_offset < 0:
    #         reason = "top_offset < 0"
    #     elif left_offset < 0:
    #         reason = "left_offset < 0"
    #     elif top_offset + size[0] > image.shape[0]:
    #         reason = "top_offset + size[0] > image.shape[0]"
    #     elif left_offset + size[1] > image.shape[1]:
    #         reason = "left_offset + size[1] > image.shape[1]"
    #     raise ValueError(f'Could not crop image\n'
    #                      f'Reason: {reason}\n'
    #                      f'Image Shape: {image.shape}\n'
    #                      f'Crop Values: [{top_offset}, {top_offset + size[0]}, '
    #                      f'{left_offset}, {left_offset + size[1]}]\n')

    if not crop:
        tx *= -1
        uncropped_image = np.zeros(image.shape, dtype=_get_dtype(image))

        if ty > 0: top_offset_u = 0
        else: top_offset_u = image.shape[0] - size[0]

        if tx > 0: left_offset_u = 0
        else: left_offset_u = image.shape[1] - size[1]

        uncropped_image[top_offset_u:top_offset_u + size[0],
                        left_offset_u:left_offset_u + size[1]] = image[top_offset:top_offset + size[0],
                                                                       left_offset:left_offset + size[1]]
        return uncropped_image
    return image[top_offset:top_offset + size[0], left_offset:left_offset + size[1]]
