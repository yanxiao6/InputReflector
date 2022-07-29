import cv2
import numpy as np

from imaugtools.helper_functions import _get_largest_rotated_rectangle, _convert_tensor_to_numpy_if_possible
from imaugtools.crop_functions import crop_around_center


def rotate_image(image, angle, crop=True):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    Parameters:
        image: Input Image
        angle: Degrees to rotate
        crop: Returns original sized image with black pixels if true
    Returns:
        image: Output Image, same size if crop is True, else cropped
    """

    # For tensor processing
    image = _convert_tensor_to_numpy_if_possible(image)

    # Get the image width and height
    # remember 0th dim is height and 1st is width
    width = image.shape[1]
    height = image.shape[0]
    image_center = (height/2, width/2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = width * 0.5
    image_h2 = height * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # For tensor processing
    if hasattr(image, 'dtype'):
        if type(image.dtype) != np.dtype:
            image = image.numpy()

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    if crop:
        image_height = image.shape[0]
        image_width = image.shape[1]
        hr, wr = _get_largest_rotated_rectangle(image_height, image_width, np.radians(angle))
        result = crop_around_center(result, (hr, wr))

    return result
