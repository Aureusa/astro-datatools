"""Standard image I/O handler, reader, and writer (PNG, JPEG, TIFF, BMP)."""
from typing import Any, Optional, Sequence, Union
import numpy as np
from PIL import Image

from .base import BaseIOHandler, BaseReader, BaseWriter
from .registry import register_handler

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]


class ImageReader(BaseReader):
    """Reader for image files into NumPy arrays."""

    def read(
        self,
        filepath: str,
        mode: str = "RGB",
        channels_first: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Read an image into a NumPy array.

        :param filepath: Path to image file.
        :param mode: Pillow color mode (e.g. 'RGB', 'L', 'RGBA'). Default is 'RGB'.
        :param channels_first: If True, returns array formatted as (C, H, W).
                               If False, returns (H, W, C) for color or (H, W) for grayscale.
        :return: np.ndarray image.
        """
        img = Image.open(filepath).convert(mode)
        arr = np.array(img)

        if mode == "L":
            if channels_first:
                return arr[np.newaxis, :, :]  # (1, H, W)
            return arr  # (H, W)

        # For RGB/RGBA: PIL gives (H, W, C)
        if channels_first and arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return arr


class ImageWriter(BaseWriter):
    """Writer for saving NumPy arrays as image files."""

    def write(
        self,
        filepath: str,
        data: np.ndarray,
        channels_first: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Write a NumPy array to an image file.

        :param filepath: Path to output image file.
        :param data: NumPy array containing image data.
        :param channels_first: Whether input array is (C, H, W). Auto-detected if None.
        :param kwargs: Additional arguments passed to PIL.Image.save.
        """
        arr = np.asarray(data)

        # Handle float arrays (normalize to 0-255 if needed, or cast)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Determine if channels first
        if channels_first is True or (channels_first is None and arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]):
            if arr.shape[0] == 1:
                img_arr = arr[0]
                mode = "L"
            else:
                img_arr = np.transpose(arr, (1, 2, 0))
                mode = "RGB" if arr.shape[0] == 3 else "RGBA"
        else:
            img_arr = arr
            if arr.ndim == 2:
                mode = "L"
            elif arr.ndim == 3 and arr.shape[2] == 3:
                mode = "RGB"
            elif arr.ndim == 3 and arr.shape[2] == 4:
                mode = "RGBA"
            else:
                mode = None

        if mode:
            img = Image.fromarray(img_arr).convert(mode)
        else:
            img = Image.fromarray(img_arr)
        img.save(filepath, **kwargs)



# Backward compatibility aliases
PngReader = ImageReader
PngWriter = ImageWriter


@register_handler(IMAGE_EXTENSIONS)
class ImageIO(BaseIOHandler):
    """Handler for standard raster image formats."""

    def __init__(self):
        self.reader = ImageReader()
        self.writer = ImageWriter()
