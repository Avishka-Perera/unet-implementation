import numpy as np
from scipy.ndimage import map_coordinates


# TODO: use the warping object
def elastic_deformation(image, sigma=10, alpha=30, points=3):
    """Applies elastic deformation to an image.

    Args:
        image: The input image as a NumPy array.
        sigma: Standard deviation of the Gaussian displacement distribution.
        alpha: Maximum displacement distance (in pixels).
        points: Number of grid points along each dimension (default: 3).

    Returns:
        The deformed image as a NumPy array.
    """

    shape = image.shape[:2]  # Extract image dimensions

    # Create a displacement grid with shape (points, points, 2)
    grid_x, grid_y = np.mgrid[0:points, 0:points]
    grid_x = (grid_x.astype(np.float32) / (points - 1)) * shape[1]
    grid_y = (grid_y.astype(np.float32) / (points - 1)) * shape[0]
    print(grid_x)

    # Generate random displacement vectors within a specified range
    displacement_x = np.random.randn(*grid_x.shape) * sigma
    displacement_y = np.random.randn(*grid_y.shape) * sigma
    displacement = np.stack([displacement_x, displacement_y], axis=-1)
    displacement *= alpha / np.max(displacement)  # Normalize for control

    # Apply bicubic interpolation to compute displacements for all pixels
    map_x, map_y = map_coordinates(displacement, (grid_y, grid_x), order=3)
    deformed_image = map_coordinates(image, (map_y, map_x), order=3).reshape(shape)

    return deformed_image
