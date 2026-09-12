"""
A fits file Writer. It opens a fits file.
It works with astropy.io.fits and inherits from BaseIO. The objects that it returns are
HDUList objects from astropy.io.fits. Their structure is as follows:
FITS file
 ├── HDU 0 (Primary HDU)
 │    ├── header
 │    └── data
 ├── HDU 1 (Extension)
 │    ├── header
 │    └── data
 ├── HDU 2 (Extension)
 │    └── ...

Also includes writers for HDF5, NumPy .npy, and PNG files.
"""
"""Writer classes and backward compatibility aliases."""
"""Writer classes and backward compatibility aliases."""
from .base import BaseWriter
from .fits import FitsWriter, get_fits_data, get_fits_header
from .hdf5 import HDF5Writer
from .image import ImageWriter, PngWriter
from .numpy_io import NpyWriter

WRITERS = {
    ".fits": FitsWriter(),
    ".hdf5": HDF5Writer(),
    ".npy": NpyWriter(),
    ".png": PngWriter(),
}


__all__ = [
    "BaseWriter",
    "FitsWriter",
    "HDF5Writer",
    "NpyWriter",
    "PngWriter",
    "ImageWriter",
    "WRITERS",
    "get_fits_header",
    "get_fits_data",
]

