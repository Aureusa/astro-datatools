import numpy as np


def embed_in_background(image: np.ndarray, background: np.ndarray, embed_type: str = "default") -> np.ndarray:
    """
    Embed an image into a background.

    :param image: Input image array.
    :type image: np.ndarray
    :param background: Background image array.
    :type background: np.ndarray
    :param embed_type: Type of embedding to perform.
    Currently supported types are:
        "default": Sum the image and background arrays.
        "fill": Embed the image into the background, keeping non-zero pixels from the image.
                Effectively replaces the background pixels with the image pixels wherever the image is non-zero.
    :type embed_type: str
    :return: Image embedded into the background.
    :rtype: np.ndarray
    """
    if image.shape != background.shape:
        raise ValueError("Image and background must have the same shape.")

    if embed_type == "default":
        return image + background
    elif embed_type == "fill":
        return np.where(image != 0, image, background)
    else:
        raise ValueError(f"Unsupported embedding type: {embed_type}")
    