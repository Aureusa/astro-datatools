import numpy as np
from itertools import combinations
from scipy.ndimage import label

import cProfile
import pstats
from pstats import SortKey


class PrecomputeProposals:
    def __init__(self, seg_map: np.ndarray, max_islands: int = 10):
        """
        Initialize precomputed proposals generator.
        
        :param seg_map: Binary segmentation map
        :param max_islands: Maximum number of islands to keep (keeps largest by area).
                            Prevents combinatorial explosion.
        """
        self.binary = np.clip(seg_map, 0, 1) # Convert to binary mask
        self.cc_map, self.num_islands = label(self.binary)
        self.max_islands = max_islands

    def _per_island_properties(self):
        if self.num_islands == 0:
            return []
        
        # Get all coordinates at once
        ys, xs = np.where(self.cc_map > 0)
        labels = self.cc_map[ys, xs]
        
        # Sort by label
        sort_idx = np.argsort(labels)
        xs_sorted = xs[sort_idx]
        ys_sorted = ys[sort_idx]
        labels_sorted = labels[sort_idx]
        
        # Find split indices
        unique_labels, split_idx = np.unique(labels_sorted, return_index=True)
        
        # Use reduceat for min/max operations
        xmins = np.minimum.reduceat(xs_sorted, split_idx)
        xmaxs = np.maximum.reduceat(xs_sorted, split_idx)
        ymins = np.minimum.reduceat(ys_sorted, split_idx)
        ymaxs = np.maximum.reduceat(ys_sorted, split_idx)
        
        # Get areas using bincount
        areas = np.bincount(labels)[unique_labels]
        
        # Build islands list
        islands = [
            {
                "id": int(unique_labels[i]),
                "area": int(areas[i]),
                "xmin": int(xmins[i]),
                "ymin": int(ymins[i]),
                "xmax": int(xmaxs[i]),
                "ymax": int(ymaxs[i]),
            }
            for i in range(len(unique_labels))
        ]
        
        # Sort by area and return top max_islands
        islands.sort(key=lambda x: x["area"], reverse=True)
        return islands[:self.max_islands]


    def precompute(self, return_scores: bool = True):
        islands = self._per_island_properties()
    
        if not islands:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)
            return (boxes, scores) if return_scores else boxes

        n = len(islands)

        # Per-island arrays
        xmin = np.array([i["xmin"] for i in islands], dtype=np.float32)
        ymin = np.array([i["ymin"] for i in islands], dtype=np.float32)
        xmax = np.array([i["xmax"] for i in islands], dtype=np.float32)
        ymax = np.array([i["ymax"] for i in islands], dtype=np.float32)
        areas = np.array([i["area"] for i in islands], dtype=np.float32)

        num_combinations = 2**n - 1
        boxes = np.empty((num_combinations, 4), dtype=np.float32)
        scores = np.empty(num_combinations, dtype=np.float32)

        idx = 0
        for r in range(1, n + 1):
            for combo in combinations(range(n), r):
                c = np.array(combo, dtype=np.int32)
                boxes[idx, 0] = xmin[c].min()
                boxes[idx, 1] = ymin[c].min()
                boxes[idx, 2] = xmax[c].max()
                boxes[idx, 3] = ymax[c].max()
                scores[idx] = areas[c].sum()
                idx += 1

        if return_scores and scores.max() > 0:
            scores /= scores.max()

        return (boxes, scores) if return_scores else boxes
