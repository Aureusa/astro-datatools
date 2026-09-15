import math
from typing import Union
import numpy as np
from scipy.ndimage import rotate, zoom


def crop(
        image: np.ndarray,
        size: int,
        *,
        crop_type: str = "center",
        loc: Union[tuple[int, int], None] = None,
        seed: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None
    ) -> np.ndarray:
    """
    Crop an image to a square of the requested size.

    :param image: Input image array. Assumes that the last two dimensions
    correspond to the spatial height and width.
    :type image: np.ndarray
    :param size: Size of the output square.
    :type size: int
    :param crop_type: Type of crop. Available options:
        - "center": Crop the center of the image.

        - "custom": Crop at a specific location provided by the ``loc`` parameter.

            ┌─────────────────────────┐
            │                         │
            │   loc ●┌──────────┐     │
            │       │           │     │
            │       │  crop     │     │
            │       │  region   │     │
            │       └───────────┘     │
            │                         │
            └─────────────────────────┘

        - "random": Crop a random region of the image.

            ┌─────────────────────────┐
            │                         │
            │       ┌──────────┐      │
            │       │  random  │      │
            │       │  region  │      │
            │       └──────────┘      │
            │                         │
            └─────────────────────────┘
    :type crop_type: str
    :param loc: Top-left corner of the crop when ``crop_type`` is "custom". Ignored for center crop.
    :type loc: tuple[int, int] | None
    :param seed: Random seed used for random crops. Ignored for other crop types.
    :type seed: int | None
    :param rng: Random number generator used for random crops. Ignored for other crop types.
    :type rng: np.random.Generator | None
    :return: Cropped image.
    :rtype: np.ndarray
    :raises ValueError: If the requested crop is larger than the input image.
    """
    h, w = image.shape[-2:]

    if h < size or w < size:
        raise ValueError(
            f"Image ({h}x{w}) is smaller than crop size {size}"
        )

    top = (h - size) // 2
    left = (w - size) // 2

    if crop_type == "center":
        return image[..., top:top + size, left:left + size]
    elif crop_type == "custom":
        if loc is None:
            raise ValueError("For custom crop, 'loc' must be provided")
        top, left = loc
        return image[..., top:top + size, left:left + size]
    elif crop_type == "random":
        return _random_crop(image, size, rng=rng, seed=seed)
    else:
        raise ValueError(f"Unknown crop_type: {crop_type}")


def _random_crop(
    image: np.ndarray,
    size: int,
    rng: Union[np.random.Generator, None] = None,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """
    Randomly crop a square region from an image.

    The crop location is sampled uniformly from all valid positions.

    :param image: Input image array. Assumes that the last two dimensions
    correspond to the spatial height and width.
    :type image: np.ndarray
    :param size: Size of the output square.
    :type size: int
    :param rng: Random number generator used to sample the crop location. If ``None``,
    a default generator is created.
    :type rng: np.random.Generator | None
    :return: Randomly cropped image.
    :rtype: np.ndarray
    :raises ValueError: If the requested crop is larger than the input image.
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    h, w = image.shape[-2:]

    if h < size or w < size:
        raise ValueError(
            f"Image ({h}x{w}) is smaller than crop size {size}"
        )

    top = rng.integers(0, h - size + 1)
    left = rng.integers(0, w - size + 1)

    return image[..., top:top + size, left:left + size]


def resize(
    image: np.ndarray,
    size: int,
    order: int = 3,
) -> np.ndarray:
    """
    Resize an image to a square spatial size.

    The channel dimension, if present, is preserved and each channel is
    resized independently.

    :param image: Input image array. Assumes that the last two dimensions
    correspond to the spatial height and width.
    :type image: np.ndarray
    :param size: Output height and width.
    :type size: int
    :param order: Interpolation order used by ``scipy.ndimage.zoom``. Higher values
    correspond to more accurate but slower interpolation. Default is 3.
    :type order: int
    :return: Resized image.
    :rtype: np.ndarray
    """
    h, w = image.shape[-2:]

    zoom_factors = (size / h, size / w)

    if image.ndim == 2:
        return zoom(image, zoom_factors, order=order)

    if image.ndim == 3:
        return zoom(
            image,
            (1.0, size / h, size / w),
            order=order,
        )

    raise ValueError(
        "Expected image with shape (H, W) or (C, H, W), "
        f"got {image.shape}"
    )


def rotation(
    image: np.ndarray,
    angle: float,
    size: Union[int, None] = None,
    order: int = 3,
) -> np.ndarray:
    """
    Rotate an image by a specified angle.

    The image is padded using reflection before rotation so that pixels near
    the corners do not expose a constant or zero-valued border. By default,
    the result is cropped back to the original spatial dimensions.

    :param image: Input image array. Assumes that the last two dimensions
        correspond to the spatial height and width.
    :type image: np.ndarray
    :param angle: Rotation angle in degrees. Positive values correspond to
        counter-clockwise rotation, while negative values correspond to
        clockwise rotation.
    :type angle: float
    :param size: If specified, the output image is center-cropped to this
        square size after rotation. If ``None``, the original spatial
        dimensions are restored.
    :type size: int | None
    :param order: Interpolation order used by ``scipy.ndimage.rotate``.
        Higher values correspond to more accurate but slower interpolation.
        Default is 3.
    :type order: int
    :return: Rotated image.
    :rtype: np.ndarray
    """
    h, w = image.shape[-2:]

    # Avoid unnecessary interpolation for rotations that leave the image
    # unchanged. This also prevents tiny floating-point artifacts.
    if angle % 360 == 0:
        if size is not None:
            return crop(image, size)
        return image.copy()

    # Pad sufficiently to ensure that the rotated corners do not expose
    # the original image boundary.
    pad = int(
        math.ceil(
            (math.sqrt(h**2 + w**2) - min(h, w)) / 2
        )
    ) + 1

    padded = np.pad(
        image,
        [(0, 0)] * (image.ndim - 2) + [(pad, pad), (pad, pad)],
        mode="reflect",
    )

    rotated = rotate(
        padded,
        angle=angle,
        axes=(-2, -1),
        reshape=False,
        order=order,
        mode="reflect",
    )

    if size is not None:
        return crop(rotated, size)

    # Restore the original H x W dimensions.
    top = (rotated.shape[-2] - h) // 2
    left = (rotated.shape[-1] - w) // 2

    return rotated[
        ...,
        top:top + h,
        left:left + w,
    ]


def hflip(
    image: np.ndarray,
) -> np.ndarray:
    """
    Flip an image horizontally.

    :param image: Input image array. Assumes that the last two dimensions
    correspond to the spatial height and width.
    :type image: np.ndarray
    :return: Horizontally flipped image.
    :rtype: np.ndarray
    """
    return np.flip(image, axis=-1).copy()


def vflip(
    image: np.ndarray,
) -> np.ndarray:
    """
    Flip an image vertically.

    :param image: Input image array. Assumes that the last two dimensions
    correspond to the spatial height and width.
    :type image: np.ndarray
    :return: Vertically flipped image.
    :rtype: np.ndarray
    """
    return np.flip(image, axis=-2).copy()
