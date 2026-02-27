import numpy as np
from itertools import combinations
from scipy.ndimage import label


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

        # Precompute aggregate box/area for every non-empty subset mask using DP.
        # This avoids repeated min/max/sum over arrays for each combination.
        subset_xmin = np.empty(num_combinations + 1, dtype=np.float32)
        subset_ymin = np.empty(num_combinations + 1, dtype=np.float32)
        subset_xmax = np.empty(num_combinations + 1, dtype=np.float32)
        subset_ymax = np.empty(num_combinations + 1, dtype=np.float32)
        subset_area = np.empty(num_combinations + 1, dtype=np.float32)

        for mask in range(1, num_combinations + 1):
            lsb = mask & -mask
            bit_idx = lsb.bit_length() - 1
            prev_mask = mask ^ lsb

            if prev_mask == 0:
                subset_xmin[mask] = xmin[bit_idx]
                subset_ymin[mask] = ymin[bit_idx]
                subset_xmax[mask] = xmax[bit_idx]
                subset_ymax[mask] = ymax[bit_idx]
                subset_area[mask] = areas[bit_idx]
            else:
                subset_xmin[mask] = min(subset_xmin[prev_mask], xmin[bit_idx])
                subset_ymin[mask] = min(subset_ymin[prev_mask], ymin[bit_idx])
                subset_xmax[mask] = max(subset_xmax[prev_mask], xmax[bit_idx])
                subset_ymax[mask] = max(subset_ymax[prev_mask], ymax[bit_idx])
                subset_area[mask] = subset_area[prev_mask] + areas[bit_idx]

        idx = 0
        for r in range(1, n + 1):
            for combo in combinations(range(n), r):
                mask = 0
                for bit_idx in combo:
                    mask |= (1 << bit_idx)

                boxes[idx, 0] = subset_xmin[mask]
                boxes[idx, 1] = subset_ymin[mask]
                boxes[idx, 2] = subset_xmax[mask]
                boxes[idx, 3] = subset_ymax[mask]
                scores[idx] = subset_area[mask]
                idx += 1

        if return_scores and scores.max() > 0:
            scores /= scores.max()

        return (boxes, scores) if return_scores else boxes
