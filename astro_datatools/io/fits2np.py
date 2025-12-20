from .fits_reader import FitsReaderIO
import numpy as np

class Fits2NpyIO(FitsReaderIO):
    """Class for converting FITS files to NumPy .npy files."""

    def write(self, filepath: str, data) -> None:
        """
        Convert FITS data to a NumPy array and save it as a .npy file.
        This method assumes that the FITS data contains image data in the first HDU.

        :param filepath: Path to the .npy file to write.
        :type filepath: str
        :param data: HDUList object containing the FITS data to convert.
        :type data: astropy.io.fits.HDUList
        """
        image_data = data[0].data  # Assuming we want the primary HDU
        np.save(filepath, image_data)
        