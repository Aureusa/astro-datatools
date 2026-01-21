import numpy as np
from skimage.segmentation import watershed


class Segment:
    """
    Segment a region based on positions and thresholding. Uses watershed segmentation.
    """
    def __init__(self, positions: list[tuple], nr_sigmas: int = 5, rms: float = 0.1*1e-3):
        """
        Initialize the Segment with positions, number of sigmas, and RMS noise level.

        :param positions: List of (x, y) (width, height) tuples indicating positions of interest.
        :type positions: list[tuple]
        :param nr_sigmas: Number of sigmas above the RMS to consider for segmentation.
        :type nr_sigmas: int
        :param rms: RMS noise level of the data.
        :type rms: float
        """
        self.positions = positions
        self.nr_sigmas = nr_sigmas
        self.rms = rms

    def get_segmentation(self, data: np.ndarray) -> np.ndarray:
        """
        Generate a segmentation map for the given data.
        Contours are created using watershed segmentation.

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: 2D numpy array representing the segmentation map.
        :rtype: np.ndarray
        """
        threshold = self.nr_sigmas * self.rms
        
        # Mask the region above the threshold
        mask = data >= threshold

        # Create markers for watershed - mark all positions at once
        markers = np.zeros_like(data, dtype=int)
        valid_positions = [(x, y) for x, y in self.positions 
                           if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]]
        if valid_positions:
            xs, ys = zip(*valid_positions)
            markers[list(ys), list(xs)] = 1
        
        markers[~mask] = 0  # Background

        # Apply watershed segmentation once with all markers
        labels = watershed(-data, markers, mask=mask)
        return labels
    

class SegmentationMap:
    """
    Class to handle multiple segments and create a full segmentation map.
    """
    def __init__(self, seg_dict: dict[str, Segment], find_grg: bool = True):
        """
        Initialize the SegmentationMap with a dictionary of segments.

        :param seg_dict: Dictionary mapping keys to Segment objects.
        :type seg_dict: dict[str, Segment]
        :param find_grg: Whether to identify and rename the giant radio galaxy (GRG) segment.
        It is assumed to be the biggest segment containing the central pixel.
        :type find_grg: bool
        """
        self.seg_dict = seg_dict
        self.find_grg = find_grg

    def get_full_segmentation(self, data: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """
        Generate a full segmentation map for the given data.
        Resolves overlaps by assigning contested pixels to the larger segment.

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: Tuple containing the segmentation map and a dictionary mapping keys to labels.
        :rtype: tuple[np.ndarray, dict[str, int]]
        """
        segmentation_mapping = {}
        segment_masks = {}
        segment_sizes = {}
        
        # First pass: get all segments and their sizes
        label = 1
        for key, segment in self.seg_dict.items():
            seg = segment.get_segmentation(data)
            segment_masks[key] = seg
            segment_sizes[key] = np.sum(seg > 0)  # Count non-zero pixels
            segmentation_mapping[key] = label
            label += 1
        
        # Second pass: build final segmentation map, resolving conflicts
        seg_map = np.zeros_like(data, dtype=int)
        overlap_count = np.zeros_like(data, dtype=int)  # Track how many segments claim each pixel
        
        # Count overlaps
        for seg in segment_masks.values():
            overlap_count += (seg > 0).astype(int)
        
        # Assign pixels, prioritizing larger segments for contested regions
        # Sort segments by size (largest first)
        sorted_keys = sorted(segment_sizes.keys(), key=lambda k: segment_sizes[k], reverse=True)
        
        for key in sorted_keys:
            seg = segment_masks[key]
            label = segmentation_mapping[key]
            
            # Get pixels belonging to this segment
            seg_pixels = seg > 0
            
            # Only assign pixels that haven't been claimed yet
            unclaimed = seg_map == 0
            seg_map[seg_pixels & unclaimed] = label

        if self.find_grg:
            segmentation_mapping = self._find_grg_segment_and_update_mapping(
                seg_map, segmentation_mapping, sorted_keys
            )
        
        return seg_map, segmentation_mapping

    def _find_grg_segment_and_update_mapping(
            self,
            seg_map: np.ndarray,
            segmentation_mapping: dict[str, int],
            sorted_keys: list
        ) -> dict[str, int]:
        """
        Identify the segment corresponding to the giant radio galaxy (GRG)
        based on the central pixel location. Also update the segmentation mapping
        to rename the identified segment as "GRG-<original_key>".
        Assumes the GRG is the LARGEST segment containing the central pixel.

        :param seg_map: 2D numpy array representing the segmentation map.
        :param segmentation_mapping: Dictionary mapping keys to labels.
        :param sorted_keys: List of segment keys sorted by size (largest first).
        :return: Updated segmentation mapping.
        """
        datashape = seg_map.shape[-1]
        center_x = datashape // 2
        center_y = center_x

        # Iterate through segments from largest to smallest
        for key in sorted_keys:
            label = segmentation_mapping[key]
            positions = np.argwhere(seg_map == label)

            if len(positions) == 0: # Safety first
                continue

            # Find bounding box of the segment
            min_row, min_col = positions.min(axis=0)
            max_row, max_col = positions.max(axis=0)
            if (min_row <= center_y <= max_row) and (min_col <= center_x <= max_col):
                # Remove old key and add new GRG-prefixed key
                del segmentation_mapping[key]
                segmentation_mapping[f"GRG-{key}"] = label
                break
        
        return segmentation_mapping
    