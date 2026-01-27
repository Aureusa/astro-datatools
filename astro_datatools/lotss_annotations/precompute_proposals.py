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
        islands = []

        for k in range(1, self.num_islands + 1):
            ys, xs = np.where(self.cc_map == k)
            area = len(xs)

            box = [xs.min(), ys.min(), xs.max(), ys.max()]
            centroid = (xs.mean(), ys.mean())

            islands.append({
                "id": k,
                "box": box,
                "area": area,
                "centroid": centroid,
                "xs": xs,  # Cache pixel positions to avoid re-computing
                "ys": ys   # Cache pixel positions to avoid re-computing
            })
        
        # Sort by area (largest first) and keep only top max_islands
        islands.sort(key=lambda x: x["area"], reverse=True)
        islands = islands[:self.max_islands]
        
        return islands
    
    def precompute(self, return_scores: bool = True):
        """
        Generate all possible combinations of islands as proposal boxes.
        For n islands, generates 2^n - 1 proposals.
        
        Returns boxes in Detectron2 format: [x1, y1, x2, y2]
        
        :param return_scores: If True, also returns objectness scores based on area.
        :type return_scores: bool
        :return: Tuple of (boxes, scores) if return_scores=True, else just boxes.
                 boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2] format
                 scores: np.ndarray of shape (N,) with objectness scores
        :rtype: tuple or np.ndarray
        """
        islands = self._per_island_properties()
        
        # Handle empty case
        if len(islands) == 0:
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            scores = np.array([], dtype=np.float32)
            return (boxes, scores) if return_scores else boxes
        
        # Pre-allocate arrays: 2^n - 1 total combinations
        num_combinations = 2**len(islands) - 1
        boxes = np.empty((num_combinations, 4), dtype=np.float32)
        scores = np.empty(num_combinations, dtype=np.float32)
        
        idx = 0
        # Generate all combinations (1 to n islands)
        for r in range(1, len(islands) + 1):
            for combo in combinations(islands, r):
                # Use cached positions instead of calling np.where again
                all_xs = np.concatenate([island["xs"] for island in combo])
                all_ys = np.concatenate([island["ys"] for island in combo])
                
                # Box in [x1, y1, x2, y2] format (Detectron2 standard)
                boxes[idx] = [all_xs.min(), all_ys.min(), all_xs.max(), all_ys.max()]
                
                # Objectness score: sum of areas
                scores[idx] = sum(island["area"] for island in combo)
                idx += 1
        
        if return_scores:
            # Normalize scores to [0, 1] range
            if scores.max() > 0:
                scores = scores / scores.max()
            print(f"Generated {len(boxes)} proposal boxes.")
            return boxes, scores
        
        return boxes