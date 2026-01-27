import numpy as np

from .segmentation import Segment


class GRGFullAnnotation:
    """
    Specialized SegmentationMap for Giant Radio Galaxy (GRG) full annotations.
    Inherits from SegmentationMap and can include additional GRG-specific methods if needed.
    """
    def __init__(self, seg_dict: dict[str, Segment], data: np.ndarray):
        """
        Initialize the GRGFullAnnotation with a dictionary of segments.

        :param seg_dict: Dictionary where keys are segment identifiers and values are Segment objects.
        :type seg_dict: dict[str, Segment]
        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        """
        self.seg_dict = seg_dict
        self.data = data

    def get_annotation(self) -> tuple[np.ndarray, dict[str, int]]:
        """
        Generate the GRG segmentation map and bounding box for the given data.
        It first annotates all segments and then identifies the GRG segment based
        on these criteria:
        
          - It contains the central pixel of the data.
         - If no segment contains the center, the segment with the largest number
         of positions/components is chosen.
         - If multiple segments satisfy the above, the largest by area is chosen.
    
        After identifying the GRG segment, its binary mask and bounding box are extracted.

        :return: Tuple containing the GRG binary segmentation map and its bounding box.
        :rtype: tuple[np.ndarray, dict[str, int]]
        """
        seg_map, bounding_boxes, segmentation_mapping, sorted_keys = self._annotate_all()

        grg_seg, grg_bbox = self._identify_grg_annotations(
            seg_map, bounding_boxes, segmentation_mapping, sorted_keys
        )
        return grg_seg, grg_bbox, seg_map
    
    def _annotate_all(self) -> dict[str, np.ndarray]:
        """
        Generate segmentation maps and bounding boxes for all segments in the segmentation map.

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: Dictionary mapping segment keys to their binary segmentation maps.
        :rtype: dict[str, np.ndarray]
        """
        segmentation_mapping = {}
        segment_masks = {}
        segment_sizes = {}
        bounding_boxes = {}
        
        # First pass: get all segments and their sizes
        label = 1
        for key, segment in self.seg_dict.items():
            seg = segment.get_segmentation(self.data)
            segment_masks[key] = seg
            segment_sizes[key] = np.sum(seg > 0)  # Count non-zero pixels
            segmentation_mapping[key] = label
            label += 1
        
        # Second pass: build final segmentation map, resolving conflicts
        seg_map = np.zeros_like(self.data, dtype=int)
        
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
            assigned_pixels = seg_pixels & unclaimed
            seg_map[assigned_pixels] = label

            # Generate bounding box from assigned pixels (more efficient than argwhere)
            if not assigned_pixels.any():
                continue
                
            rows, cols = np.where(assigned_pixels)
            bounding_boxes[key] = {
                'top': rows.max(),
                'bottom': rows.min(),
                'left': cols.min(),
                'right': cols.max()
            }

        return seg_map, bounding_boxes, segmentation_mapping, sorted_keys

    def _identify_grg_annotations(
            self,
            seg_map: np.ndarray,
            bounding_boxes: dict[str, dict[str, int]],
            segmentation_mapping: dict[str, int],
            sorted_keys: list
        ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Identify the GRG segment and extract its binary mask and bounding box.

        :param seg_map: 2D numpy array representing the segmentation map.
        :type seg_map: np.ndarray
        :param bounding_boxes: Dictionary mapping segment keys to their bounding box coordinates.
        :type bounding_boxes: dict[str, dict[str, int]]
        :param segmentation_mapping: Dictionary mapping keys to labels.
        :type segmentation_mapping: dict[str, int]
        :param sorted_keys: List of segment keys sorted by size (largest first).
        :type sorted_keys: list
        :return: Tuple containing the GRG binary segmentation map and its bounding box.
        :rtype: tuple[np.ndarray, dict[str, int]]
        """
        datashape = seg_map.shape[-1]
        center_x = datashape // 2
        center_y = center_x

        # Iterate through segments from largest to smallest
        grg_key = None
        for key in sorted_keys:
            # Skip segments that have no pixels in the final segmentation map
            if key not in bounding_boxes:
                continue

            # Find bounding box of the segment
            min_row, min_col = bounding_boxes[key]['bottom'], bounding_boxes[key]['left']
            max_row, max_col = bounding_boxes[key]['top'], bounding_boxes[key]['right']
            if (min_row <= center_y <= max_row) and (min_col <= center_x <= max_col):
                grg_key = key

        # If no segment contains the center, pick the segment with the biggest amount
        # of positions = components as GRG (only consider segments with bounding boxes)
        if grg_key is None:
            max_positions = 0
            for key, segment in self.seg_dict.items():
                # Only consider segments that have pixels in the final segmentation map
                if key not in bounding_boxes:
                    continue
                curr_positions = len(segment.positions)
                if curr_positions > max_positions:
                    max_positions = curr_positions
                    grg_key = key

        # Final check    
        if grg_key is None:
            return None, None
        
        # Now we have identified the GRG segment, extraxt a binary mask for it and the bbox
        grg_label = segmentation_mapping[grg_key]
        grg_seg = (seg_map == grg_label).astype(int)
        grg_bbox = bounding_boxes[grg_key]
        return grg_seg, grg_bbox
    