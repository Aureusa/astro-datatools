from astropy.io import fits
from astropy.io.fits import HDUList
from PIL import Image
import numpy as np

from .fits_reader import FitsReaderIO


class Fits2PngIO(FitsReaderIO):
    """Class for converting FITS files to PNG images."""

    def write(self, filepath: str, data: HDUList) -> None:
        """
        Convert FITS data to PNG and save it. Assumes the FITS data contains image data
        in the first HDU. Also assumes the image data is in a format compatible with RGB - 
        meaning 3 channels. It converts the data to 8-bit per channel.

        :param filepath: Path to the PNG file to write.
        :type filepath: str
        :param data: HDUList object containing the FITS data to convert.
        :type data: astropy.io.fits.HDUList
        """
        image_data = data[0].data  # Assuming we want the primary HDU

        # Convert to 8-bit per channel if necessary
        r = image_data[0].astype(np.uint8)
        g = image_data[1].astype(np.uint8)
        b = image_data[2].astype(np.uint8)

        rgb = np.stack([r, g, b], axis=-1)
        Image.fromarray(rgb, mode='RGB').save(filepath)
