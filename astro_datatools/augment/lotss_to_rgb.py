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
from ..transforms.stretch import sqrt_stretch, asinh_stretch

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
        Supports both single images (H, W) -> (C, H, W) and batches (B, H, W) -> (B, C, H, W).
        """
        # Pre-calculate thresholds to avoid repeated multiplications
        threshold_3sigma = 3 * self.rms_noise
        threshold_5sigma = 5 * self.rms_noise
        
        # Determine if batched or single image
        if data.ndim == 2:
            # Single image (H, W)
            channel_1 = sqrt_stretch(data) if not self.asinh_stretch else \
                asinh_stretch(data, a=threshold_5sigma)
            channel_2 = (data >= threshold_3sigma).astype(np.float32)
            channel_3 = (data >= threshold_5sigma).astype(np.float32)
            
            # Stack channels to create RGB image (C, H, W)
            rgb_data = np.stack([channel_1, channel_2, channel_3], axis=0)
            
        elif data.ndim == 3:
            # Batched images (B, H, W)
            channel_1 = sqrt_stretch(data) if not self.asinh_stretch else \
                asinh_stretch(data, a=threshold_5sigma)
            channel_2 = (data >= threshold_3sigma).astype(np.float32)
            channel_3 = (data >= threshold_5sigma).astype(np.float32)
            
            # Stack channels to create RGB images (B, C, H, W)
            rgb_data = np.stack([channel_1, channel_2, channel_3], axis=1)
            
        else:
            raise ValueError(f"Expected 2D (H, W) or 3D (B, H, W) input, got shape {data.shape}")
        
        return rgb_data
        