"""I/O package for astronomical and scientific data formats."""
from .astro import AstroIO
from .base import BaseIOHandler, BaseReader, BaseWriter
from .fits import FitsIO, FitsReader, FitsWriter, get_fits_data, get_fits_header
from .hdf5 import HDF5IO, HDF5Reader, HDF5Writer
from .image import ImageIO, ImageReader, ImageWriter, PngReader, PngWriter
from .numpy_io import NpyIO, NpyReader, NpyWriter
from .registry import (
    DEFAULT_REGISTRY,
    IORegistry,
    register_handler,
    register_reader,
    register_writer,
)

__all__ = [
    "AstroIO",
    "BaseReader",
    "BaseWriter",
    "BaseIOHandler",
    "FitsIO",
    "FitsReader",
    "FitsWriter",
    "HDF5IO",
    "HDF5Reader",
    "HDF5Writer",
    "NpyIO",
    "NpyReader",
    "NpyWriter",
    "ImageIO",
    "ImageReader",
    "ImageWriter",
    "PngReader",
    "PngWriter",
    "IORegistry",
    "DEFAULT_REGISTRY",
    "register_reader",
    "register_writer",
    "register_handler",
    "get_fits_header",
    "get_fits_data",
]
