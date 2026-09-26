import math
from typing import Tuple, Union
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


def rebin_matrix(n_in: int, n_out: int) -> np.ndarray:
    """
    Build the 1D area-weighted rebinning matrix between two regular grids.

    Both grids are assumed to cover the same extent (the unit interval), with
    ``n_in`` and ``n_out`` equal-width pixels respectively. Entry ``[i, j]`` is
    the fraction of output pixel ``i`` covered by input pixel ``j``, so every
    row sums to one and ``W @ x`` is the area-weighted *mean* of ``x`` inside
    each output pixel. Works for any, including non-integer, size ratio.

    :param n_in: Number of input pixels along the axis.
    :type n_in: int
    :param n_out: Number of output pixels along the axis.
    :type n_out: int
    :return: Matrix of shape ``(n_out, n_in)`` whose rows sum to one.
    :rtype: np.ndarray
    :raises ValueError: If either size is smaller than 1.
    """
    n_in, n_out = int(n_in), int(n_out)
    if n_in < 1 or n_out < 1:
        raise ValueError(f"Grid sizes must be >= 1, got n_in={n_in}, n_out={n_out}")

    edges_in = np.arange(n_in + 1) / n_in
    edges_out = np.arange(n_out + 1) / n_out
    overlap = np.clip(
        np.minimum(edges_out[1:, None], edges_in[None, 1:])
        - np.maximum(edges_out[:-1, None], edges_in[None, :-1]),
        0,
        None,
    )
    return overlap / overlap.sum(axis=1, keepdims=True)


def rebin(
    image: np.ndarray,
    out_shape: Union[int, Tuple[int, int]],
    conserve: str = "surface_brightness",
) -> np.ndarray:
    """
    Resample an image onto a coarser or finer pixel grid by exact area weighting.

    Each output pixel is the area-weighted combination of the input pixels it
    overlaps, with both grids spanning the same field of view. Unlike spline
    interpolation (e.g. :func:`resize`), this does not alias when shrinking an
    image by large or non-integer factors: every input pixel contributes to
    the output in proportion to its overlap, so small bright clumps are neither
    dropped nor boosted. For an integer shrink factor it reduces to an exact
    block average (or block sum).

    The operation is separable and implemented as ``Wy @ image @ Wx.T`` with
    :func:`rebin_matrix`, broadcasting over all leading axes.

    :param image: Input image. The last two dimensions are the spatial height
        and width, e.g. ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)``.
    :type image: np.ndarray
    :param out_shape: Output spatial size, either an int (square output) or
        ``(height, width)``.
    :type out_shape: int | tuple[int, int]
    :param conserve: Quantity preserved by the resampling:

        - ``"surface_brightness"`` (default): the mean pixel value is
          preserved. Appropriate for images whose pixel values are intensities
          or display units (brightness per pixel area).
        - ``"flux"``: the total sum is preserved. Appropriate for flux maps
          where each pixel holds the flux collected in that pixel; values are
          multiplied by the ratio of output to input pixel area.
    :type conserve: str
    :return: Rebinned image of shape ``image.shape[:-2] + out_shape`` (float).
    :rtype: np.ndarray
    :raises ValueError: If ``conserve`` is unknown, ``out_shape`` is invalid or
        the image has fewer than two dimensions.
    """
    if conserve not in ("surface_brightness", "flux"):
        raise ValueError(
            f"Unknown conserve mode {conserve!r}; expected 'surface_brightness' or 'flux'"
        )
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError(f"Expected an image with at least 2 dimensions, got {image.shape}")

    if isinstance(out_shape, (int, np.integer)):
        out_shape = (int(out_shape), int(out_shape))
    ny_out, nx_out = (int(n) for n in out_shape)
    ny_in, nx_in = image.shape[-2:]

    if (ny_out, nx_out) == (ny_in, nx_in):
        return image.astype(np.result_type(image.dtype, np.float32), copy=True)

    wy = rebin_matrix(ny_in, ny_out)
    wx = rebin_matrix(nx_in, nx_out)
    rebinned = wy @ image @ wx.T

    if conserve == "flux":
        rebinned = rebinned * ((ny_in * nx_in) / (ny_out * nx_out))

    return rebinned


def crop_or_pad(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]],
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Centre-crop or symmetrically pad an image to a fixed spatial size.

    No resampling is performed. Each spatial axis is handled independently:
    if it is longer than the target it is centre-cropped with the same
    convention as :func:`crop` (``start = (n - size) // 2``); if it is shorter
    it is padded with ``before = (size - n) // 2`` and
    ``after = size - n - before`` pixels of ``pad_value``. The image centre is
    therefore kept at the centre of the output (to within one pixel).

    :param image: Input image. The last two dimensions are the spatial height
        and width.
    :type image: np.ndarray
    :param size: Output spatial size, an int (square) or ``(height, width)``.
    :type size: int | tuple[int, int]
    :param pad_value: Constant used for padded pixels. Default is 0.
    :type pad_value: float
    :return: Image with spatial shape ``(height, width)``.
    :rtype: np.ndarray
    """
    if isinstance(size, (int, np.integer)):
        size = (int(size), int(size))
    target = tuple(int(s) for s in size)
    if any(s < 1 for s in target):
        raise ValueError(f"Output size must be >= 1, got {size}")

    image = np.asarray(image)
    slices = [slice(None)] * image.ndim
    pad_width = [(0, 0)] * image.ndim

    for axis, s in zip((-2, -1), target):
        n = image.shape[axis]
        if n > s:
            start = (n - s) // 2
            slices[axis] = slice(start, start + s)
        elif n < s:
            before = (s - n) // 2
            pad_width[axis] = (before, s - n - before)

    out = image[tuple(slices)]
    if any(p != (0, 0) for p in pad_width):
        out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)
    else:
        out = out.copy()
    return out


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
