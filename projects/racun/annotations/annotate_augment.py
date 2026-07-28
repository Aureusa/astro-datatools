import numpy as np
from scipy.ndimage import label as ndi_label

from astro_datatools.augment import RotateAugment, LotssToRGBAugment
from .precompute_proposals import PrecomputeProposals as proposals_generator


def _to_serializable_scalar(value):
    """Convert numpy scalars to plain Python types for JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_grouping_metadata(
        candidates_keys: list[str],
        candidates_values: list,
        angles: list[int],
        valid_positions_list: list[np.ndarray],
        kept_indices_list: list[np.ndarray],
    ) -> dict:
    """Create a JSON-serializable per-angle grouping schema using local IDs only."""
    source_names = np.asarray(candidates_values[0], dtype=object)
    component_names = np.asarray(candidates_values[1], dtype=object)
    physical_keys = candidates_keys[3:]
    physical_arrays = {
        key: np.asarray(candidates_values[i + 3]) for i, key in enumerate(physical_keys)
    }

    num_components = int(source_names.shape[0])

    angles_payload = []
    for angle_idx, angle in enumerate(angles):
        kept_indices = np.asarray(kept_indices_list[angle_idx], dtype=int)
        valid_positions = np.asarray(valid_positions_list[angle_idx])

        source_id_map = {}
        angle_components = []
        source_to_component_ids = {}
        for local_id, component_id in enumerate(kept_indices):
            if component_id >= num_components:
                continue

            source_name = str(source_names[component_id])
            if source_name not in source_id_map:
                source_id_map[source_name] = len(source_id_map)
            source_id = int(source_id_map[source_name])

            component_entry = {
                "component_id": int(local_id),
                "source_id": source_id,
                "source_name": source_name,
                "component_name": str(component_names[component_id]),
                "xy": [
                    int(_to_serializable_scalar(valid_positions[local_id, 0])) if local_id < len(valid_positions) else -1,
                    int(_to_serializable_scalar(valid_positions[local_id, 1])) if local_id < len(valid_positions) else -1,
                ],
                "physical": {},
            }
            for key in physical_keys:
                arr = physical_arrays[key]
                component_entry["physical"][key] = _to_serializable_scalar(arr[component_id])
            angle_components.append(component_entry)
            source_to_component_ids.setdefault(source_id, []).append(int(local_id))

        angle_sources = []
        for source_name, source_id in source_id_map.items():
            component_ids = source_to_component_ids.get(source_id, [])
            if component_ids:
                angle_sources.append({
                    "source_id": source_id,
                    "source_name": source_name,
                    "component_ids": component_ids,
                })

        angles_payload.append({
            "angle": int(angle),
            "components": angle_components,
            "sources": angle_sources,
        })

    return {
        "schema": "b2s-grouping-v2-local-only",
        "angles": angles_payload,
    }


def _segmentation_from_positions_via_connected_components(
        data: np.ndarray,
        positions: list[tuple[int, int]],
        indeces_to_keep: list[int],
        fluxes: list[float],
        max_islands: int,
        nr_sigmas: int,
        rms: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast equivalent for Segment(...).get_segmentation(data) when all markers share
    the same label (which is the case in Segment.get_segmentation).

    It keeps connected components of the threshold mask that contain at least one
    valid marker position.
    """
    empty_positions = np.empty((0, 2), dtype=np.int32)
    empty_indices = np.empty((0,), dtype=np.int32)

    if not positions:
        return np.zeros_like(data, dtype=data.dtype), empty_positions, empty_indices

    # Make sure to remove the fluxes that are outside of the
    # cutout due to rotation and cropping
    fluxes = np.array(fluxes)[indeces_to_keep]

    # TODO: Time this, it might be slow!
    num_fluxes = len(fluxes)
    if num_fluxes > (max_islands if max_islands is not None else num_fluxes + 1):
        # Remove the smallest fluxes to keep only max_islands
        sorted_indices = np.argsort(fluxes)
        constrained_sorted_indices = sorted_indices[-max_islands:]
        positions_array = np.asarray(positions, dtype=np.int32)[constrained_sorted_indices]
        fluxes = fluxes[constrained_sorted_indices]
        indeces_to_keep = indeces_to_keep[constrained_sorted_indices]
    else:
        positions_array = np.asarray(positions, dtype=np.int32)

    threshold = nr_sigmas * rms
    mask = data >= threshold
    if not mask.any():
        return np.zeros_like(data, dtype=data.dtype), empty_positions, empty_indices

    cc_map, _ = ndi_label(mask)
    if cc_map.max() == 0:
        return np.zeros_like(data, dtype=data.dtype), empty_positions, empty_indices

    valid = (
        (positions_array[:, 0] >= 0)
        & (positions_array[:, 0] < data.shape[1])
        & (positions_array[:, 1] >= 0)
        & (positions_array[:, 1] < data.shape[0])
    )
    if not np.any(valid):
        return np.zeros_like(data, dtype=data.dtype), empty_positions, empty_indices

    valid_positions = positions_array[valid]
    valid_indices_to_keep = np.asarray(indeces_to_keep, dtype=np.int32)[valid]
    marker_components = cc_map[valid_positions[:, 1], valid_positions[:, 0]]
    marker_valid = marker_components > 0
    valid_positions = valid_positions[marker_valid]
    valid_indices_to_keep = valid_indices_to_keep[marker_valid]
    marker_components = marker_components[marker_valid]
    if marker_components.size == 0:
        return np.zeros_like(data, dtype=data.dtype), empty_positions, empty_indices

    keep_components = np.unique(marker_components)
    max_label = int(cc_map.max())
    keep_lut = np.zeros(max_label + 1, dtype=bool)
    keep_lut[keep_components] = True
    seg = keep_lut[cc_map]
    return seg.astype(data.dtype, copy=False), valid_positions, valid_indices_to_keep

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
    indeces_to_keep_list = []
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
        indeces_to_keep = np.where(valid_mask)[0]
        indeces_to_keep_list.append(indeces_to_keep)
    
    return rotated_xy_list, indeces_to_keep_list

def annotate_and_augment(
        data: np.ndarray,
        candidates: dict[str, list],
        angles: list[int],
        labels: dict[str, int] = None,
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
        valid_positions_list,
        gt_instance_bboxes,
        gt_instance_masks,
        gt_instance_category_ids,
        gt_instance_positions,
        augmented_proposals,
        proposal_scores,
        grouping_metadata
    )
    :rtype: tuple
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

    candidates_keys = list(candidates.keys()) # Get the keys of the candidates dict to maintain the same order
    candidates_values = list(candidates.values()) # Get the values of the candidates dict in the same order as the keys

    source_components_arr = np.array(candidates_values[:2]) # holds the source names and component names - shape (2, num_components)

    # Next: Augment with rotation
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
    
    # Rotate all component positions
    all_component_positions = candidates.get("xy_list", [])  # Get the list of (x, y) positions from candidates
    rotated_all_component_positions, original_indices_to_keep_after_rotation = rotate_xy_list_of_points(
        all_component_positions, angles, crop_dims, original_h, original_w
    ) # shape (num_angles, num_components) - list of lists of (x, y) positions for each angle
    
    # Generate segmentation maps for each angle for all components.
    augmented_seg_map = np.zeros_like(rotated_data, dtype=rotated_data.dtype)
    original_indices_to_keep_after_segmentation_list = []
    # This list contains the positions used to create the segmentation map for each angle,
    # which are the positions that are both within the crop after rotation and within the
    # segmentation mask after thresholding. We will use these positions to compute the
    # physics-aware features later on, as they correspond to the components that are actually
    # considered in the segmentation and proposal generation process for each angle.
    valid_positions_list = []
    for i in range(rotated_data.shape[0]):
        curr_data = rotated_data[i]
        augmented_seg_map[i], valid_positions, original_indices_to_keep_after_segmentation = _segmentation_from_positions_via_connected_components(
            data=curr_data,
            positions=rotated_all_component_positions[i],
            indeces_to_keep=original_indices_to_keep_after_rotation[i],
            fluxes=candidates.get("total_flux", []),
            max_islands=max_precomputed_islands,
            nr_sigmas=nr_sigmas,
            rms=rms,
        )
        original_indices_to_keep_after_segmentation_list.append(original_indices_to_keep_after_segmentation)
        valid_positions_list.append(valid_positions)

    num_angles = augmented_seg_map.shape[0]
    
    # Generate region proposals for the Masked RCNN model for each angle
    augmented_proposals = []
    proposal_scores = []
    gt_instance_bboxes = []
    gt_instance_masks = []
    gt_instance_category_ids = []
    gt_instance_positions = []
    for i in range(num_angles):
        angle_indices = np.asarray(original_indices_to_keep_after_segmentation_list[i], dtype=np.int32)
        angle_source_labels = np.asarray(source_components_arr[0])[angle_indices]

        (
            proposed_boxes,
            angle_proposal_scores,
            angle_gt_instance_bboxes,
            angle_gt_instance_masks,
            angle_gt_instance_category_ids,
            angle_gt_instance_positions,
        ) = proposals_generator(
            augmented_seg_map[i], max_islands=max_precomputed_islands
        ).precompute(
            return_scores=True,
            return_ground_truth=True,
            return_instance_targets=True,
            labels=labels,
            component_positions=np.asarray(valid_positions_list[i], dtype=np.int32),
            component_source_labels=angle_source_labels,
        )
        augmented_proposals.append(proposed_boxes)
        proposal_scores.append(angle_proposal_scores)
        gt_instance_bboxes.append(angle_gt_instance_bboxes)
        gt_instance_masks.append(angle_gt_instance_masks)
        gt_instance_category_ids.append(angle_gt_instance_category_ids)
        gt_instance_positions.append(angle_gt_instance_positions)

    grouping_metadata = _build_grouping_metadata(
        candidates_keys=candidates_keys,
        candidates_values=candidates_values,
        angles=angles,
        valid_positions_list=valid_positions_list,
        kept_indices_list=original_indices_to_keep_after_segmentation_list,
    )

    # We convert the rotated_data to augmented into RGB image
    lotss_to_rgba = LotssToRGBAugment(rms_noise=rms, asinh_stretch=asinh_stretch)
    augmented_data = lotss_to_rgba.augment(rotated_data)

    # Finally
    return (
        augmented_data,
        valid_positions_list,
        gt_instance_bboxes,
        gt_instance_masks,
        gt_instance_category_ids,
        gt_instance_positions,
        augmented_proposals,
        proposal_scores,
        grouping_metadata,
    )
