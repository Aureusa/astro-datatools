"""
HDUList structure is as follows:
FITS file
 ├── HDU 0 (Primary HDU)
 │    ├── header
 │    └── data
 ├── HDU 1 (Extension)
 │    ├── header
 │    └── data
 ├── HDU 2 (Extension)
 │    └── ...
"""
"""Reader classes and backward compatibility aliases."""
"""Reader classes and backward compatibility aliases."""
from .base import BaseReader
from .fits import FitsReader, get_fits_data, get_fits_header
from .hdf5 import HDF5Reader
from .image import ImageReader, PngReader
from .numpy_io import NpyReader

READERS = {
    ".fits": FitsReader(),
    ".hdf5": HDF5Reader(),
    ".npy": NpyReader(),
    ".png": PngReader(),
}


__all__ = [
    "BaseReader",
    "FitsReader",
    "HDF5Reader",
    "NpyReader",
    "PngReader",
    "ImageReader",
    "READERS",
    "get_fits_header",
    "get_fits_data",
]

