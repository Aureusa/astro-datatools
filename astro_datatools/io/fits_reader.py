"""
A fits file reader. It opens a fits file.
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
"""
from astropy.io import fits
from astropy.io.fits import HDUList

from .base import BaseIO


class FitsReaderIO(BaseIO):
    """Class for reading from FITS files."""

    def read(self, filepath: str) -> HDUList:
        """Read data from a FITS file.

        :param filepath: Path to the FITS file to read.
        :type filepath: str
        :return: HDUList object containing the FITS data.
        :rtype: astropy.io.fits.HDUList
        """
        with fits.open(filepath) as hdul:
            data = hdul.copy()
        return data

    def write(self, filepath: str, data: HDUList) -> None:
        """Overwrite in child class, not used in this reader."""
        raise NotImplementedError(f"Write method is not implemented in {self.__class__.__name__}.")
