import numpy as np
from skimage.transform import resize
from scipy.ndimage import map_coordinates


def warp_image(image, flow):
    height, width = image.shape[0], image.shape[1]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Apply the flow field to the coordinates
    y_warped = y + flow[..., 0]
    x_warped = x + flow[..., 1]

    # Clip the coordinates to stay within the image boundaries
    y_warped = np.clip(y_warped, 0, height - 1)
    x_warped = np.clip(x_warped, 0, width - 1)

    # Interpolate the values using map_coordinates
    warped_image = map_coordinates(
        image[..., 0], (y_warped, x_warped), order=1, mode="reflect"
    )

    # Reshape the result to match the input shape
    warped_image = warped_image.reshape((height, width, 1))

    return warped_image


def get_9_pt_flow(shape, std=10):
    original_array = np.random.randn(3, 3, 2) * std
    pad_size = 1
    padded_array = np.pad(
        original_array,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    target_shape = (*shape, 2)
    resized_array = resize(padded_array, target_shape, mode="edge", anti_aliasing=True)
    return resized_array


def random_warp(img, std=10):
    shape = img.shape[-2:]
    flow = get_9_pt_flow(shape, std=std)
    warped_img = warp_image(img, flow)
    return warped_img
