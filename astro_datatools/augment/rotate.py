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

        # Rotate all angles - this is already reasonably efficient
        augmented_data = [
            rotate(data, angle, axes=self.height_and_width_axes, reshape=False)
            for angle in self.angles
        ]
        augmented_data = np.stack(augmented_data, axis=0) # Shape: (num_angles, ...)

        # Crop - pre-calculate target size once
        if self.dynamic_cropping:
            new_sizes = [self.largest_rotated_rect(w, h, angle) for angle in self.angles]
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

    def augment_bbox(self, bbox: dict[str, int], original_h: int, original_w: int) -> list[dict[str, int]]:
        """
        Apply rotation augmentation to a bounding box.

        :param bbox: Bounding box with keys 'top', 'bottom', 'left', 'right'
        :param original_h: Original image height before rotation
        :param original_w: Original image width before rotation
        :return: List of rotated bounding boxes for each angle
        """
        # Pre-calculate crop dimensions for each angle to avoid redundant calculation
        crop_dims = []
        for angle in self.angles:
            if self.dynamic_cropping:
                cropped_w, cropped_h = self.largest_rotated_rect(original_w, original_h, angle)
            elif self.specific_crop_size is not None:
                cropped_w, cropped_h = self.specific_crop_size
            else:
                cropped_w, cropped_h = original_w, original_h
            crop_dims.append((cropped_w, cropped_h))
        
        rotated_bboxes = [
            self._rotate_bbox(bbox, angle, original_h, original_w, crop_h, crop_w)
            for angle, (crop_w, crop_h) in zip(self.angles, crop_dims)
        ]
        
        return rotated_bboxes
    
    def augment_proposal_boxes(self, boxes: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
        """
        Apply rotation augmentation to multiple boxes in Detectron2 format (vectorized).
        
        :param boxes: Array of boxes in [x1, y1, x2, y2] format, shape (N, 4)
        :type boxes: np.ndarray
        :param original_h: Original image height before rotation
        :type original_h: int
        :param original_w: Original image width before rotation
        :type original_w: int
        :return: Rotated boxes for all angles, shape (num_angles, N, 4)
        :rtype: np.ndarray
        """
        num_boxes = boxes.shape[0]
        num_angles = len(self.angles)
        
        # Pre-calculate crop dimensions for each angle
        crop_dims = []
        for angle in self.angles:
            if self.dynamic_cropping:
                cropped_w, cropped_h = self.largest_rotated_rect(original_w, original_h, angle)
            elif self.specific_crop_size is not None:
                cropped_w, cropped_h = self.specific_crop_size
            else:
                cropped_w, cropped_h = original_w, original_h
            crop_dims.append((cropped_w, cropped_h))
        
        # Get all 4 corners of all boxes: shape (N, 4, 2)
        # Scipy rotates in (row, col) = (y, x) space, so store as (y, x)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        corners = np.stack([
            np.stack([y2, x1], axis=1),  # (y, x) for bottom-left
            np.stack([y2, x2], axis=1),  # (y, x) for bottom-right
            np.stack([y1, x2], axis=1),  # (y, x) for top-right
            np.stack([y1, x1], axis=1),  # (y, x) for top-left
        ], axis=1)  # Shape: (N, 4, 2) in (y, x) format
        
        # Center in (y, x) format to match scipy
        center = np.array([(original_h - 1) / 2, (original_w - 1) / 2])
        all_rotated_boxes = []
        
        # Process each angle
        for angle, (crop_w, crop_h) in zip(self.angles, crop_dims):
            angle_rad = np.radians(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Counter-clockwise rotation matrix (matches scipy.ndimage.rotate)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            
            # Rotate all corners of all boxes at once
            corners_centered = corners - center  # Shape: (N, 4, 2)
            corners_rotated = corners_centered @ rotation_matrix.T  # Shape: (N, 4, 2)
            corners_rotated += center
            
            # Adjust for cropping
            crop_offset = np.array([(original_w - crop_w) / 2, (original_h - crop_h) / 2])
            corners_rotated -= crop_offset
            
            # Get axis-aligned bounding boxes from rotated corners (in y,x format)
            min_coords = corners_rotated.min(axis=1)  # Shape: (N, 2) as (y_min, x_min)
            max_coords = corners_rotated.max(axis=1)  # Shape: (N, 2) as (y_max, x_max)
            
            # Clip to image bounds
            min_coords = np.maximum(0, np.floor(min_coords)).astype(np.float32)
            max_coords = np.minimum([crop_h, crop_w], np.ceil(max_coords)).astype(np.float32)
            
            # Convert from (y, x) back to (x, y) and combine to [x1, y1, x2, y2] format
            rotated_boxes = np.stack([
                min_coords[:, 1],  # x1 from x_min
                min_coords[:, 0],  # y1 from y_min
                max_coords[:, 1],  # x2 from x_max
                max_coords[:, 0]   # y2 from y_max
            ], axis=1)  # Shape: (N, 4)
            all_rotated_boxes.append(rotated_boxes)
        
        # Stack to (num_angles, N, 4)
        return np.stack(all_rotated_boxes, axis=0)

    def _rotate_bbox(
            self,
            bbox: dict[str, int],
            angle: float,
            original_h: int,
            original_w: int, 
            cropped_h: int,
            cropped_w: int
        ) -> dict[str, int]:
        """
        Rotate a bounding box and adjust for cropping.
        
        :param bbox: Bounding box with keys 'top', 'bottom', 'left', 'right'
        :param angle: Rotation angle in degrees
        :param original_h: Original image height before rotation
        :param original_w: Original image width before rotation
        :param cropped_h: Height after rotation and cropping
        :param cropped_w: Width after rotation and cropping
        :return: Rotated and cropped bounding box
        """
        # Get bbox corners in (y, x) format to match scipy rotation
        corners = np.array([
            [bbox['bottom'], bbox['left']],   # (y, x) bottom-left
            [bbox['bottom'], bbox['right']],  # (y, x) bottom-right
            [bbox['top'], bbox['right']],     # (y, x) top-right
            [bbox['top'], bbox['left']]       # (y, x) top-left
        ])
        
        # Rotate around image center in (y, x) format
        # Center matches scipy.ndimage.rotate: (height-1)/2, (width-1)/2
        center = np.array([(original_h - 1) / 2, (original_w - 1) / 2])
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Counter-clockwise rotation matrix (matches scipy.ndimage.rotate)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Translate to origin, rotate, translate back
        corners_centered = corners - center
        corners_rotated = corners_centered @ rotation_matrix.T
        corners_rotated += center
        
        # Adjust for cropping (center crop)
        crop_offset_y = (original_h - cropped_h) / 2
        crop_offset_x = (original_w - cropped_w) / 2
        corners_rotated -= np.array([crop_offset_y, crop_offset_x])
        
        # Get new axis-aligned bbox from (y, x) corners
        min_y, min_x = corners_rotated.min(axis=0)
        max_y, max_x = corners_rotated.max(axis=0)
        
        # Clip to image bounds
        min_x = max(0, int(np.floor(min_x)))
        min_y = max(0, int(np.floor(min_y)))
        max_x = min(cropped_w, int(np.ceil(max_x)))
        max_y = min(cropped_h, int(np.ceil(max_y)))
        
        return {
            'left': min_x,
            'right': max_x,
            'bottom': min_y,
            'top': max_y
        }

    def largest_rotated_rect(self, w: int, h: int, angle: float) -> tuple[int, int]:
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
