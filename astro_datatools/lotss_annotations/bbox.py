import numpy as np

from .segmentation import Segment, SegmentationMap

class BBox:
    """
    Generate bounding boxes from segmentation maps.
    """
    def __init__(self, seg_dict: dict[str, Segment], find_grg: bool = True):
        """
        Initialize the BBox with a dictionary of segments.
        
        :param seg_dict: Dictionary where keys are segment identifiers and values are Segment objects.
        :type seg_dict: dict[str, Segment]
        :param find_grg: Whether to identify and rename the giant radio galaxy (GRG) segment.
        It is assumed to be the biggest segment containing the central pixel.
        :type find_grg: bool
        """
        self._segmentation_map = SegmentationMap(seg_dict, find_grg=find_grg)

    def get_bounding_boxes(self, data: np.ndarray) -> dict[str, dict[str, int]]:
        """
        Generate bounding boxes for each segment in the segmentation map.

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: Dictionary mapping segment keys to their bounding box coordinates.
                 Each bounding box is represented as a dictionary with keys 'top', 'bottom', 'left', 'right'.
        :rtype: dict[str, dict[str, int]]
        """
        seg_map, segmentation_mapping = self._segmentation_map.get_full_segmentation(data)
        bounding_boxes = {}

        for key, label in segmentation_mapping.items():
            # Find the indices where the segmentation map equals the current label
            positions = np.argwhere(seg_map == label)

            if len(positions) > 0:
                # Get min/max coordinates
                min_row, min_col = positions.min(axis=0)
                max_row, max_col = positions.max(axis=0)
                
                bounding_boxes[key] = {
                    'top': max_row,
                    'bottom': min_row,
                    'left': min_col,
                    'right': max_col
                }
            else:
                bounding_boxes[key] = None  # No segmentation found for this key

        return bounding_boxes
    