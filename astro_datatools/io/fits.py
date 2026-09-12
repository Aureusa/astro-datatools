"""FITS I/O handler, reader, writer, and astronomy utilities."""
import os
from typing import Any, Optional, Tuple, Union

from astropy.io import fits
from astropy.io.fits import HDUList, Header
import numpy as np

from .base import BaseIOHandler, BaseReader, BaseWriter
from .registry import register_handler

FITS_EXTENSIONS = [".fits", ".fits.gz", ".fit", ".fts", ".fz"]


class FitsReader(BaseReader):
    """Reader for astronomical FITS files."""

    def read(
        self,
        filepath: str,
        memmap: bool = True,
        lazy: bool = False,
        ext: Optional[Union[int, str]] = None,
        return_header: bool = False,
        **kwargs: Any,
    ) -> Union[HDUList, np.ndarray, Tuple[np.ndarray, Header]]:
        """Read data from a FITS file.

        :param filepath: Path to the FITS file.
        :param memmap: Whether to use memory mapping. Default is True.
        :param lazy: If True, returns open HDUList without forcing memory load.
        :param ext: Specific HDU index or name to read. If specified, returns HDU data directly.
        :param return_header: If True and ext is specified, returns (data, header).
        :param kwargs: Additional arguments passed to `fits.open`.
        :return: HDUList, np.ndarray, or (data, header) tuple.
        """
        hdul = fits.open(filepath, memmap=memmap, **kwargs)

        if ext is not None:
            try:
                hdu = hdul[ext]
                data = hdu.data
                header = hdu.header
                if return_header:
                    return data, header
                return data
            finally:
                hdul.close()

        if lazy or memmap:
            return hdul

        # If not lazy and not memmap, load all data into memory and close file handle
        try:
            for hdu in hdul:
                _ = hdu.data
            return hdul
        except Exception:
            hdul.close()
            raise


class FitsWriter(BaseWriter):
    """Writer for astronomical FITS files."""

    def write(
        self,
        filepath: str,
        data: Union[HDUList, fits.PrimaryHDU, fits.ImageHDU, np.ndarray],
        header: Optional[Header] = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write data to a FITS file.

        :param filepath: Destination filepath.
        :param data: HDUList, HDU object, or numpy array.
        :param header: Optional FITS header when data is a numpy array.
        :param overwrite: Whether to overwrite existing file. Default is True.
        :param kwargs: Additional arguments passed to `writeto`.
        """
        if isinstance(data, HDUList):
            data.writeto(filepath, overwrite=overwrite, **kwargs)
        elif isinstance(data, (fits.PrimaryHDU, fits.ImageHDU, fits.BinTableHDU, fits.TableHDU)):
            hdul = HDUList([data])
            hdul.writeto(filepath, overwrite=overwrite, **kwargs)
        elif isinstance(data, np.ndarray):
            hdu = fits.PrimaryHDU(data=data, header=header)
            hdu.writeto(filepath, overwrite=overwrite, **kwargs)
        else:
            raise TypeError(
                f"Unsupported data type for FITS writer: {type(data)}. "
                f"Expected HDUList, HDU, or numpy.ndarray."
            )


@register_handler(FITS_EXTENSIONS)
class FitsIO(BaseIOHandler):
    """High-level astronomical FITS handler providing convenience methods."""

    def __init__(self):
        self.reader = FitsReader()
        self.writer = FitsWriter()

    def open(self, filepath: str, mode: str = "readonly", memmap: bool = True, **kwargs: Any) -> HDUList:
        """Open a FITS file as a context-managed HDUList.

        Usage:
            with fits_io.open('galaxy.fits') as hdul:
                data = hdul[0].data
        """
        return fits.open(filepath, mode=mode, memmap=memmap, **kwargs)

    def read_image(
        self,
        filepath: str,
        ext: Union[int, str] = 0,
        return_header: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Header]]:
        """Read 2D/3D image data directly from a FITS file.
        If ext=0 has no data (common in multi-extension FITS), falls back to ext=1.

        :param filepath: Path to FITS file.
        :param ext: HDU index or name (default is 0).
        :param return_header: If True, returns (data, header) tuple.
        :return: Numpy array or (data, header).
        """
        with fits.open(filepath, **kwargs) as hdul:
            if ext == 0 and hdul[0].data is None and len(hdul) > 1:
                target_ext = 1
            else:
                target_ext = ext
            hdu = hdul[target_ext]
            data = np.copy(hdu.data) if hdu.data is not None else None
            header = hdu.header.copy()

        if return_header:
            return data, header
        return data

    def write_image(
        self,
        filepath: str,
        data: np.ndarray,
        header: Optional[Header] = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write a numpy array image directly to a FITS file.

        :param filepath: Path to output FITS file.
        :param data: Image array.
        :param header: Optional header.
        :param overwrite: Overwrite existing file.
        """
        self.writer.write(filepath, data, header=header, overwrite=overwrite, **kwargs)

    def read_header(self, filepath: str, ext: Union[int, str] = 0, **kwargs: Any) -> Header:
        """Read only the header of a specific HDU without loading heavy data.

        :param filepath: Path to FITS file.
        :param ext: HDU index or name (default is 0).
        :return: astropy.io.fits.Header
        """
        return fits.getheader(filepath, ext=ext, **kwargs)

    def read_table(self, filepath: str, ext: Union[int, str] = 1, **kwargs: Any) -> Any:
        """Read tabular data from a FITS extension.

        :param filepath: Path to FITS file.
        :param ext: Extension index or name (default is 1).
        :return: Table data or structured array.
        """
        with fits.open(filepath, **kwargs) as hdul:
            return hdul[ext].data

    def get_header(self, hdul: HDUList, idx: Union[int, str] = 0) -> Header:
        """Get header from an already opened HDUList."""
        return hdul[idx].header

    def get_data(self, hdul: HDUList, idx: Union[int, str] = 0) -> Any:
        """Get data from an already opened HDUList."""
        return hdul[idx].data

    def info(self, filepath_or_hdul: Union[str, HDUList]) -> Any:
        """Print / return summary info of the HDUList."""
        if isinstance(filepath_or_hdul, str):
            with fits.open(filepath_or_hdul) as hdul:
                return hdul.info()
        return filepath_or_hdul.info()


# Top-level helper functions for backward compatibility & convenience
def get_fits_header(idx: Union[int, str], hdul: HDUList) -> Header:
    """Get the header of a specific HDU in the FITS file."""
    return hdul[idx].header


def get_fits_data(idx: Union[int, str], hdul: HDUList) -> Any:
    """Get the data of a specific HDU in the FITS file."""
    return hdul[idx].data
