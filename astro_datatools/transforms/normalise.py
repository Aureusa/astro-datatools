import numpy as np


def normalise(image: np.ndarray, bounds: tuple = (0, 1)):
    """
    Normalise an image to the specified bounds using min-max scaling.

    :param image: Input image array.
    :type image: np.ndarray
    :param bounds: Tuple specifying the desired range for the output image (default: (0, 1)).
    :type bounds: tuple
    :return: Normalised image array.
    :rtype: np.ndarray
    """
    # First do minmax
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) # in range [0, 1]

    # Then scale to the desired bounds
    lower, upper = bounds
    return image * (upper - lower) + lower


def zscore_normalise(image: np.ndarray):
    """
    Normalise an image using z-score normalization.

    :param image: Input image array.
    :type image: np.ndarray
    :return: Z-score normalised image array. This will have a mean of 0 and a
    standard deviation of 1, unless the standard deviation is 0,
    in which case the image will be mean-centered.
    :rtype: np.ndarray
    """
    mean_val = np.mean(image)
    std_val = np.std(image)
    return (image - mean_val) / std_val if std_val != 0 else image - mean_val
