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
    def __init__(self, seg_dict: dict[str, Segment]):
        """
        Initialize the SegmentationMap with a dictionary of segments.

        :param seg_dict: Dictionary mapping keys to Segment objects.
        :type seg_dict: dict[str, Segment]
        """
        self.seg_dict = seg_dict

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
        
        return seg_map, segmentation_mapping
    