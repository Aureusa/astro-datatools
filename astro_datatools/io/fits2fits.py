from astropy.io.fits import HDUList

from .fits_reader import FitsReaderIO


class Fits2FitsIO(FitsReaderIO):
    """Class for reading from and writing to FITS files."""

    def write(self, filepath: str, data: HDUList) -> None:
        """Write data to a FITS file.

        :param filepath: Path to the FITS file to write.
        :type filepath: str
        :param data: HDUList object containing the FITS data to write.
        :type data: astropy.io.fits.HDUList
        """
        data.writeto(filepath, overwrite=True)
