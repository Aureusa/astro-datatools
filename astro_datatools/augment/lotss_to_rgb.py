"""
In the first channel:
    - square-root stretched between 1 and 30 sigma
In the second channel:
    - all radio emission above three sigma are set to one while all radio emission below that value to zero.
In the third channel:
    - all radio emissions above five sigma are set to one and all radio emission below that value to zero.

Encoding schema adopted from: "Radio source-component association for the LOFAR Two-metre
Sky Survey with region-based convolutional neural networks" by Mostert et al. (2022) doi: https://doi.org/10.1051/0004-6361/202243478
"""
import numpy as np

from .base import BaseAugment


class LotssToRGBAugment(BaseAugment):
    """Class for converting LoTSS data to RGB format."""

    def __init__(self, rms_noise: float, asinh_stretch: bool = False):
        """
        Initialize the LotssToRGBAugment with specified RMS noise.

        :param rms_noise: RMS noise level of the radio data.
        :type rms_noise: float
        :param asinh_stretch: Whether to use asinh stretch for the first channel.
        If left False, square-root stretch is used.
        :type asinh_stretch: bool
        """
        self.rms_noise = rms_noise
        self.asinh_stretch = asinh_stretch

    def augment(self, data: np.ndarray) -> np.ndarray:
        """
        Convert LoTSS data to RGB format as described in Mostert et al. (2022).

        :param data: Input data containing 'radio_data' and 'rms_noise'.
        :type data: dict
        :return: Augmented data with 'rgb_data' key added.
        :rtype: dict
        """
        # Create RGB channels
        channel_1 = np.sqrt(np.clip(data / (30 * self.rms_noise), 0, 1)) if not self.asinh_stretch else \
            np.arcsinh(data / (30 * self.rms_noise)) / np.arcsinh(1)
        channel_2 = np.where(data >= 3 * self.rms_noise, 1.0, 0.0)
        channel_3 = np.where(data >= 5 * self.rms_noise, 1.0, 0.0)

        # Stack channels to create RGB image
        rgb_data = np.stack([channel_1, channel_2, channel_3], axis=0)
        return rgb_data
    