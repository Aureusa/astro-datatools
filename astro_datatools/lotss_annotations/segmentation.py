import numpy as np
from skimage.segmentation import watershed


class Segment:
    """
    Segment a region based on positions and thresholding. Uses watershed segmentation.
    """
    def __init__(self, positions: list[tuple], nr_sigmas: int = 5, rms: float = 0.1*1e-3):
        """
        Initialize the Segment with positions, number of sigmas, and RMS noise level.

        :param positions: List of (y, x) (heigh, width) tuples indicating positions of interest.
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

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: 2D numpy array representing the segmentation map.
        :rtype: np.ndarray
        """
        labels = None
        for position in self.positions:
            if labels is None:
                labels = self._is_within_threshold(data, position)
            else:
                labels += self._is_within_threshold(data, position)
        return labels

    def _is_within_threshold(self, data: np.ndarray, position: tuple) -> np.ndarray:
        """
        Check if the data at the given position is within the threshold.

        :param data: 2D numpy array representing the data to check.
        :type data: np.ndarray
        :param position: Tuple (y, x) indicating the position to check.
        :type position: tuple
        :return: 2D numpy array representing the segmentation labels.
        :rtype: np.ndarray
        """
        threshold = self.nr_sigmas * self.rms
        
        # Mask the region above the threshold
        mask = data >= threshold

        # Create markers for watershed
        markers = np.zeros_like(data, dtype=int)
        y, x = position
        markers[y, x] = 1  # Marker for the region of interest
        markers[~mask] = 0  # Marker for the background

        # Apply watershed segmentation
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

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: Tuple containing the segmentation map and a dictionary mapping keys to labels.
        :rtype: tuple[np.ndarray, dict[str, int]]
        """
        segmentation_mapping = {}

        seg_map = np.zeros_like(data, dtype=int)
        label = 1
        for key, segment in self.seg_dict.items():
            seg = segment.get_segmentation(data)
            seg_map += label * seg

            segmentation_mapping[key] = label
            label += 1
        return seg_map, segmentation_mapping
    