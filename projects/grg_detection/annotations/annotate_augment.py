import numpy as np
import logging

from astro_datatools.lotss_annotations.segmentation import Segment
from astro_datatools.augment import RotateAugment, LotssToRGBAugment
from .precompute_proposals import PrecomputeProposals as proposals_generator


def rotate_xy_list_of_points(
        xy_list: list[tuple[int, int]],
        angles: list[int],
        crop_dims: list[tuple[int, int]],
        original_h: int,
        original_w: int
    ) -> list[list[tuple[int, int]]]:
    """
    Rotate a list of (x, y) points for each angle.
    This is done by using the rotation matrix directly for efficiency.
    The equation used is based on the counter-clockwise rotation matrix
    to match scipy.ndimage.rotate behavior:
    A = | cos(θ)  -sin(θ) |
        | sin(θ)   cos(θ) |
    B = | x - cx |
        | y - cy |
    A * B + C = | x' |
                | y' |
    where (cx, cy) is the center of rotation and (x', y') are the rotated coordinates.

    :param xy_list: List of (x, y) tuples representing positions.
    :type xy_list: list[tuple[int, int]]
    :param angles: List of angles (in degrees) to rotate the positions.
    :type angles: list[int]
    :param crop_dims: List of (width, height) tuples representing crop dimensions for each angle
    :type crop_dims: list[tuple[int, int]]
    :param original_h: Original image height before rotation
    :type original_h: int
    :param original_w: Original image width before rotation
    :type original_w: int
    :return: List of lists of rotated positions for each angle
    :rtype: list[list[tuple[int, int]]]
    """
    if not xy_list:
        return [[] for _ in angles]
    
    # Convert positions to numpy array: shape (N, 2) in (y, x) format for scipy
    xy_array = np.array([(y, x) for x, y in xy_list])  # Convert (x, y) to (y, x)
    
    # Center in (y, x) format to match scipy
    center = np.array([(original_h - 1) / 2, (original_w - 1) / 2])
    
    rotated_xy_list = []
    for angle, (crop_w, crop_h) in zip(angles, crop_dims):
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Counter-clockwise rotation matrix (matches scipy.ndimage.rotate)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate all positions at once
        positions_centered = xy_array - center
        positions_rotated = positions_centered @ rotation_matrix.T
        positions_rotated += center
        
        # Adjust for cropping
        crop_offset = np.array([(original_h - crop_h) / 2, (original_w - crop_w) / 2])
        positions_rotated -= crop_offset
        
        # Round to integers
        positions_rotated = np.round(positions_rotated).astype(int)
        
        # Filter out positions outside crop boundaries (positions_rotated is in (y, x) format)
        valid_mask = (
            (positions_rotated[:, 0] >= 0) & (positions_rotated[:, 0] < crop_h) &
            (positions_rotated[:, 1] >= 0) & (positions_rotated[:, 1] < crop_w)
        )
        positions_rotated = positions_rotated[valid_mask]
        
        # Convert back to (x, y) tuples
        rotated_positions = [(int(x), int(y)) for y, x in positions_rotated]  # Convert (y, x) back to (x, y)
        
        rotated_xy_list.append(rotated_positions)
    
    return rotated_xy_list

def annotate_and_augment(
        data: np.ndarray,
        grg_positions: list[tuple[int, int]],
        all_component_positions: list[tuple[int, int]],
        angles: list[int],
        height_and_width_axes: tuple = (-2, -1),
        dynamic_cropping: bool = True,
        specific_crop_size: tuple[int, int] = None,
        max_precomputed_islands: int = 10,
        nr_sigmas: int = 3,
        rms: float = 0.1*1e-3,
        asinh_stretch: bool = False
    ) -> np.ndarray:
    """
    Augment astronomical data through rotation and generate corresponding annotations.
    
    This function performs data augmentation by rotating input data at specified angles,
    then generates all necessary annotations for training object detection models:
    segmentation maps, bounding boxes, and region proposals. The rotated data is
    converted to RGB format as described in Mostert et al. (2022).

    :param data: Input astronomical image data to augment.
    :type data: np.ndarray
    :param grg_positions: List of (x, y) pixel positions marking Giant
    Radio Galaxy (GRG) components.
    :type grg_positions: list[tuple[int, int]]
    :param all_component_positions: List of (x, y) pixel positions
    marking all radio components in the cutout.
    :type all_component_positions: list[tuple[int, int]]
    :param angles: List of rotation angles in degrees to apply for augmentation.
    :type angles: list[int]
    :param height_and_width_axes: Tuple specifying which axes represent (height, width).
    Defaults to (-2, -1).
    :type height_and_width_axes: tuple
    :param dynamic_cropping: Whether to dynamically crop to the largest inscribed
    rectangle after rotation. Defaults to True.
    :type dynamic_cropping: bool
    :param specific_crop_size: Specific (width, height) to crop to after rotation,
    overrides dynamic_cropping if set. Defaults to None.
    :type specific_crop_size: tuple[int, int]
    :param max_precomputed_islands: Maximum number of island regions to generate
    as proposals. Defaults to 10.
    :type max_precomputed_islands: int
    :param rms: RMS noise level for the LoTSS data.
    Defaults to 0.1*1e-3.
    :type rms: float
    :param asinh_stretch: Whether to apply asinh stretch during RGB conversion.
    Defaults to False.
    :type asinh_stretch: bool
    :return: Tuple containing (
        augmented_data,
        rotated_grg_positions,
        rotated_all_component_positions,
        augmented_grg_segm,
        augmented_bboxes,
        augmented_proposals,
        augmented_proposal_scores
    )
    :rtype: tuple[np.ndarray, list[list[tuple[int, int]]], list[list[tuple[int, int]]], 
            np.ndarray, list[list[int]], list[np.ndarray], list[np.ndarray]]
    """
    # First: Augment the data by rotating it for each angle
    original_w, original_h = data.shape[height_and_width_axes[1]], data.shape[height_and_width_axes[0]]
    rotator = RotateAugment(
        angles=angles,
        height_and_width_axes=height_and_width_axes,
        dynamic_cropping=dynamic_cropping,
        specific_crop_size=specific_crop_size
    )
    rotated_data = rotator.augment(data) # (num_angles, height, width) after rotation and cropping

    # Next: Augment the positions accordingly - needed to compute segmentation maps and bboxes
    # Pre-calculate crop dimensions for each angle
    crop_dims = []
    for angle in angles:
        if dynamic_cropping:
            cropped_w, cropped_h = rotator.largest_rotated_rect(original_w, original_h, angle)
        elif specific_crop_size is not None:
            cropped_w, cropped_h = specific_crop_size
        else:
            cropped_w, cropped_h = original_w, original_h
        crop_dims.append((cropped_w, cropped_h))
    
    # Rotate GRG and all component positions
    rotated_all_component_positions = rotate_xy_list_of_points(
        all_component_positions, angles, crop_dims, original_w, original_h
    )
    rotated_grg_positions = rotate_xy_list_of_points(
        grg_positions, angles, crop_dims, original_w, original_h
    )
    
    # Generate segmentation maps for each angle based on the rotated positions for GRG
    augmented_grg_segm = np.zeros_like(rotated_data, dtype=rotated_data.dtype)
    for i in range(rotated_data.shape[0]):
        augmented_grg_segm[i] = Segment(
            positions=rotated_grg_positions[i],
            nr_sigmas=nr_sigmas,
            rms=rms
        ).get_segmentation(rotated_data[i])

    # Now for the components segmentation maps
    augmented_seg_map = np.zeros_like(rotated_data, dtype=rotated_data.dtype)
    for i in range(rotated_data.shape[0]):
        augmented_seg_map[i] = Segment(
            positions=rotated_all_component_positions[i],
            nr_sigmas=nr_sigmas,
            rms=rms
        ).get_segmentation(rotated_data[i])

    # Generate bboxes for all angles that cover the grg_segmentation areas 
    # Get the bounding boxes for each angle by finding x1, y1, x2, y2 (x_min, y_min, x_max, y_max)
    # The final product will have shape (num_angles, 4)
    num_angles = augmented_grg_segm.shape[0]
    augmented_bboxes = []
    for i in range(num_angles):
        mask = augmented_grg_segm[i] > 0
        if mask.any():
            rows, cols = np.where(mask)
            x_min = int(cols.min())
            x_max = int(cols.max())
            y_min = int(rows.min())
            y_max = int(rows.max())
            augmented_bboxes.append([x_min, y_min, x_max, y_max])
        else:
            # No valid segmentation for this angle - append None
            augmented_bboxes.append(None)
    
    # Generate region proposals for the Masked RCNN model for each angle
    augmented_proposals = []
    augmented_proposal_scores = []
    for i in range(num_angles):
        proposed_boxes, proposal_scores = proposals_generator(
            augmented_seg_map[i], max_islands=max_precomputed_islands
        ).precompute(return_scores=True)
        augmented_proposals.append(proposed_boxes)
        augmented_proposal_scores.append(proposal_scores)

    # Finally we convert the rotated_data to augmented into RGB image
    lotss_to_rgba = LotssToRGBAugment(rms_noise=rms, asinh_stretch=asinh_stretch)
    augmented_data = lotss_to_rgba.augment(rotated_data)
    return (
        augmented_data,
        rotated_grg_positions,
        rotated_all_component_positions,
        augmented_grg_segm,
        augmented_bboxes,
        augmented_proposals,
        augmented_proposal_scores
    )
