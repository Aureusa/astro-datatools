import numpy as np

from astro_datatools.lotss_annotations.segmentation import Segment


class GRGFinder:
    """
    Identifies Giant Radio Galaxy (GRG) components from segmented astronomical data.
    
    This class processes multiple segments to identify which components belong to the
    primary GRG source versus other radio sources in the field.

    The decision criteria for identifying the GRG segment are as follows (in order of precedence):
        - The segment contains the central pixel of the data.
        - If no segment contains the center, the segment with the largest number
            of positions/components is chosen.
        - If multiple segments satisfy the above, the largest by area is chosen.
    """
    def __init__(self, seg_dict: dict[str, Segment], data: np.ndarray):
        """
        Initialize the FindGRG with a dictionary of segments and data.

        :param seg_dict: Dictionary where keys are segment identifiers and values are Segment objects.
        :type seg_dict: dict[str, Segment]
        :param data: 2D numpy array representing the astronomical image data.
        :type data: np.ndarray
        """
        self.seg_dict = seg_dict
        self.data = data

    def get_positions(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Identify GRG component positions and all component positions in the data.
        
        First segments all components, then identifies which segment represents the GRG
        based on these criteria (in order of precedence):
        
        - Contains the central pixel of the data
        - Has the largest number of positions/components (if no segment contains center)
        - Has the largest area (if multiple segments have same number of components)
    
        :return: Tuple containing (grg_positions, all_component_positions) where each is a list of (x, y) tuples.
        :rtype: tuple[list[tuple[int, int]], list[tuple[int, int]]]
        """
        seg_map, bounding_boxes, positions, sorted_keys = self._segment_and_box()

        all_component_positions = []
        for pos_list in positions.values():
            all_component_positions.extend(pos_list)

        grg_positions = self._identify_grg(
            seg_map, bounding_boxes, positions, sorted_keys
        )
        return grg_positions, all_component_positions
    
    def _segment_and_box(
            self
        ) -> tuple[np.ndarray, dict[str, dict[str, int]], dict[str, list[tuple[int, int]]], list[str]]:
        """
        Generate segmentation maps and bounding boxes for all segments.
        
        Creates a unified segmentation map by processing all segments, resolving overlaps
        by prioritizing larger segments. Also computes bounding boxes for each segment.

        :return: Tuple containing (seg_map, bounding_boxes, positions, sorted_keys).
                 seg_map is the 2D segmentation array, bounding_boxes maps segment keys to box coordinates,
                 positions maps keys to component positions, sorted_keys are segment keys by size.
        :rtype: tuple[np.ndarray, dict[str, dict[str, int]], dict[str, list[tuple[int, int]]], list[str]]
        """
        segment_masks = {}  # {key: segmentation map}
        segment_sizes = {}  # {key: size in pixels}
        bounding_boxes = {}  # {key: {top, bottom, left, right}}
        positions = {}  # {key: list of (x, y) positions}
        
        # First pass: get all segments and their sizes
        label = 1
        for key, segment in self.seg_dict.items():
            seg = segment.get_segmentation(self.data)
            segment_masks[key] = seg
            segment_sizes[key] = np.sum(seg > 0)  # Count non-zero pixels
            positions[key] = segment.positions
            label += 1
        
        # Second pass: build final segmentation map, resolving overlapping pixels
        seg_map = np.zeros_like(self.data, dtype=int)
        
        # Assign pixels, prioritizing larger segments for overlapping regions
        # Segments are already sorted by size (largest first)
        sorted_keys = sorted(segment_sizes.keys(), key=lambda k: segment_sizes[k], reverse=True)
        
        for key in sorted_keys:
            seg = segment_masks[key]
            
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
                'bottom': rows.max(),  # Maximum row index (bottom of image)
                'top': rows.min(),     # Minimum row index (top of image)
                'left': cols.min(),
                'right': cols.max()
            }

        return seg_map, bounding_boxes, positions, sorted_keys

    def _identify_grg(
            self,
            seg_map: np.ndarray,
            bounding_boxes: dict[str, dict[str, int]],
            positions: dict[str, list[tuple[int, int]]],
            sorted_keys: list
        ) -> list[tuple[int, int]]:
        """
        Identify which segment represents the GRG and return its component positions.

        :param seg_map: 2D numpy array representing the segmentation map.
        :type seg_map: np.ndarray
        :param bounding_boxes: Dictionary mapping segment keys to their bounding box coordinates.
        :type bounding_boxes: dict[str, dict[str, int]]
        :param positions: Dictionary mapping segment keys to lists of (x, y) component positions.
        :type positions: dict[str, list[tuple[int, int]]]
        :param sorted_keys: List of segment keys sorted by size in pixels (largest first).
        :type sorted_keys: list
        :return: List of (x, y) positions for the identified GRG components.
        :rtype: list[tuple[int, int]]
        """
        datashape = seg_map.shape[-1]
        center_x = datashape // 2
        center_y = center_x

        # Find the segment containing the center pixel (checking from largest to smallest)
        grg_key = None
        for key in sorted_keys:
            # Skip segments that have no pixels in the final segmentation map
            if key not in bounding_boxes:
                continue

            # Find bounding box of the segment
            min_row, min_col = bounding_boxes[key]['top'], bounding_boxes[key]['left']
            max_row, max_col = bounding_boxes[key]['bottom'], bounding_boxes[key]['right']
            if (min_row <= center_y <= max_row) and (min_col <= center_x <= max_col):
                grg_key = key

        # If no segment contains the center, pick the segment with the most
        # component positions as the GRG (only consider segments with bounding boxes)
        if grg_key is None:
            max_positions = 0
            for key in positions:
                # Only consider segments that have pixels in the final segmentation map
                if key not in bounding_boxes:
                    continue
                curr_positions = len(positions[key])
                if curr_positions > max_positions:
                    max_positions = curr_positions
                    grg_key = key

        # Final validation
        if grg_key is None:
            return None, None
        
        # Return the positions of the identified GRG segment
        grg_positions = positions[grg_key]
        return grg_positions
    