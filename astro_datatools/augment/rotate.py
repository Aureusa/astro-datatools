import numpy as np
from scipy.ndimage import rotate
import logging

from .base import BaseAugment


logger = logging.getLogger(__name__)


class RotateAugment(BaseAugment):
    """Class for rotating data as an augmentation technique."""

    def __init__(
            self,
            angles: list[int],
            height_and_width_axes: tuple = (-2, -1),
            dynamic_cropping: bool = True,
            specific_crop_size: tuple[int, int] = None
        ):
        """
        Initialize the RotateAugment with specified angles.

        :param angles: List of angles (in degrees) to rotate the data.
        :type angles: list[int]
        :param height_and_width_axes: Tuple specifying the axes corresponding to height and width.
        :type height_and_width_axes: tuple
        :param dynamic_cropping: Whether to crop dynamically based on rotation.
        When set to True, the largest possible rectangle without black borders will be used.
        If False, no cropping is applied unless specific_crop_size is provided and empty regions
        are padded with zeros.
        :type dynamic_cropping: bool
        :param specific_crop_size: Specific size (width, height) to crop to after rotation.
        :type specific_crop_size: tuple[int, int]
        """
        self.angles = angles
        self.height_and_width_axes = height_and_width_axes

        # Copping configurations
        self.dynamic_cropping = dynamic_cropping
        self.specific_crop_size = specific_crop_size

        if self.dynamic_cropping and self.specific_crop_size is not None:
            logger.error("Cannot set both dynamic_cropping and specific_crop_size.")
            raise ValueError("Cannot set both dynamic_cropping and specific_crop_size.")

        info = f"RotateAugment initialized with angles: {self.angles}"
        info += f"\nDynamic cropping: {self.dynamic_cropping}" if self.dynamic_cropping else ""
        info += f"\nSpecific crop size: {self.specific_crop_size}" if self.specific_crop_size is not None else ""
        logger.info(info)

    def augment(self, data: np.ndarray) -> np.ndarray:
        """
        Apply rotation augmentation to the data.
        It works with data of arbitrary shape, rotating along specified height and width axes.

        :param data: Input data to be augmented.
        :type data: np.ndarray
        :return: Augmented data with rotations applied.
        :rtype: np.ndarray
        """
        h = data.shape[self.height_and_width_axes[0]]
        w = data.shape[self.height_and_width_axes[1]]

        logger.debug(f"Original data shape: {data.shape}")

        augmented_data = [
            rotate(data, angle, axes=self.height_and_width_axes, reshape=False)
            for angle in self.angles
        ]
        augmented_data = np.stack(augmented_data, axis=0) # Shape: (num_angles, ...)

        # Crop
        if self.dynamic_cropping:
            new_sizes = [self._largest_rotated_rect(w, h, angle) for angle in self.angles]

            # Find the smallest width and height to crop all to the same size
            new_w = min(size[0] for size in new_sizes)
            new_h = min(size[1] for size in new_sizes)
        elif self.specific_crop_size is not None:
            new_w, new_h = self.specific_crop_size
        else:
            # No crop
            return augmented_data
        
        augmented_data = self._crop_center(augmented_data, new_w, new_h)

        logger.debug(f"Augmented data shape after rotation and cropping: {augmented_data.shape}")
        return augmented_data

    def _largest_rotated_rect(self, w: int, h: int, angle: float) -> tuple[int, int]:
        """
        Calculate the largest rectangle that fits inside a rotated image without black borders.
        
        :param w: original width
        :param h: original height
        :param angle: rotation angle in degrees
        :return: (new_width, new_height)
        """
        angle_rad = np.radians(angle)
        
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)
        
        # Calculate the largest rectangle
        sin_a = abs(np.sin(angle_rad))
        cos_a = abs(np.cos(angle_rad))
        
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            # Half-constrained case
            x = 0.5 * side_short
            wr = x / sin_a if sin_a > 0 else side_long
            hr = x / cos_a if cos_a > 0 else side_long
        else:
            # Fully constrained case
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a
        
        return int(wr), int(hr)
    
    def _crop_center(self, img: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
        """
        Crop the center of an image.
        
        :param img: Input image array.
        :type img: np.ndarray
        :param crop_w: Width of the crop.
        :type crop_w: int
        :param crop_h: Height of the crop.
        :type crop_h: int
        :return: Center-cropped image.
        :rtype: np.ndarray
        """
        h, w = img.shape[-2:]
        start_y = h // 2 - crop_h // 2
        start_x = w // 2 - crop_w // 2
        return img[..., start_y:start_y + crop_h, start_x:start_x + crop_w]
