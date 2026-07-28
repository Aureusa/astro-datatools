import numpy as np
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
        
        # Sort by area (descending). Selection to max_islands is handled in precompute,
        # where we can prioritize islands that contain provided components.
        islands.sort(key=lambda x: x["area"], reverse=True)
        return islands


    def precompute(
        self,
        return_scores: bool = True,
        return_ground_truth: bool = False,
        return_instance_targets: bool = False,
        labels: dict[str, int] = None,
        component_positions: np.ndarray = None,
        component_source_labels: np.ndarray = None,
    ):
        # Instance category encoding when return_ground_truth=True:
        # 1 -> valid single-component source (SCS)
        # 2 -> valid multi-component source (MCS)
        SCS_LABEL = labels["scs"] if labels and "scs" in labels else 1
        MCS_LABEL = labels["mcs"] if labels and "mcs" in labels else 2

        all_islands = self._per_island_properties()
        islands = all_islands
    
        if return_instance_targets and not return_ground_truth:
            raise ValueError("return_instance_targets=True requires return_ground_truth=True.")

        if not islands:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)
            if return_ground_truth:
                if return_instance_targets:
                    gt_instance_bboxes = np.zeros((0, 4), dtype=np.float32)
                    gt_instance_masks = []
                    gt_instance_category_ids = np.zeros((0,), dtype=np.int32)
                    gt_instance_positions = []
                    if return_scores:
                        return (
                            boxes,
                            scores,
                            gt_instance_bboxes,
                            gt_instance_masks,
                            gt_instance_category_ids,
                            gt_instance_positions,
                        )
                    return (
                        boxes,
                        gt_instance_bboxes,
                        gt_instance_masks,
                        gt_instance_category_ids,
                        gt_instance_positions,
                    )
            return (boxes, scores) if return_scores else boxes

        if return_ground_truth:
            if component_positions is None or component_source_labels is None:
                raise ValueError(
                    "component_positions and component_source_labels are required when return_ground_truth=True."
                )

            component_positions = np.asarray(component_positions, dtype=np.int32)
            component_source_labels = np.asarray(component_source_labels)
            if component_positions.ndim != 2 or component_positions.shape[1] != 2:
                raise ValueError(
                    f"component_positions must have shape (num_components, 2), got {component_positions.shape}."
                )
            if component_source_labels.shape[0] != component_positions.shape[0]:
                raise ValueError(
                    "component_source_labels length must match component_positions length. "
                    f"Got {component_source_labels.shape[0]} and {component_positions.shape[0]}."
                )

            num_components = int(component_positions.shape[0])

            # Keep islands that contain at least one component marker first.
            # This avoids dropping true-source islands when max_islands truncation is active.
            if self.max_islands is not None and len(all_islands) > self.max_islands:
                in_bounds = (
                    (component_positions[:, 0] >= 0)
                    & (component_positions[:, 0] < self.cc_map.shape[1])
                    & (component_positions[:, 1] >= 0)
                    & (component_positions[:, 1] < self.cc_map.shape[0])
                )
                if np.any(in_bounds):
                    valid_positions = component_positions[in_bounds]
                    component_island_ids = set(
                        int(v) for v in np.unique(self.cc_map[valid_positions[:, 1], valid_positions[:, 0]]) if int(v) > 0
                    )
                else:
                    component_island_ids = set()

                # Keep ALL islands with components, then fill up to max_islands with non-component islands.
                # This ensures no component loses its island due to truncation.
                prioritized = [island for island in all_islands if int(island["id"]) in component_island_ids]
                remaining = [island for island in all_islands if int(island["id"]) not in component_island_ids]
                max_remaining = max(0, self.max_islands - len(prioritized))
                islands = prioritized + remaining[:max_remaining]
            elif self.max_islands is not None:
                islands = all_islands[: self.max_islands]

            # Map connected-component labels to island indices in the selected island list.
            island_ids = np.array([island["id"] for island in islands], dtype=np.int32)
            island_id_to_idx = {int(island_id): idx for idx, island_id in enumerate(island_ids)}

            island_component_bits = [0] * len(islands)
            for comp_idx, (x_pos, y_pos) in enumerate(component_positions):
                if y_pos < 0 or x_pos < 0 or y_pos >= self.cc_map.shape[0] or x_pos >= self.cc_map.shape[1]:
                    continue
                cc_label = int(self.cc_map[y_pos, x_pos])
                if cc_label in island_id_to_idx:
                    island_component_bits[island_id_to_idx[cc_label]] |= (1 << comp_idx)

            # Encode sources to integer IDs and precompute full source-component bitmasks.
            _, source_ids = np.unique(component_source_labels, return_inverse=True)
            source_component_bits = {}
            for comp_idx, source_id in enumerate(source_ids):
                source_id = int(source_id)
                source_component_bits[source_id] = source_component_bits.get(source_id, 0) | (1 << comp_idx)
        elif self.max_islands is not None:
            islands = all_islands[: self.max_islands]

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

        if return_ground_truth:
            subset_component_bits = [0] * (num_combinations + 1)
            if return_instance_targets:
                gt_instance_bboxes = []
                gt_instance_masks = []
                gt_instance_category_ids = []
                gt_instance_positions = []

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

            if return_ground_truth:
                subset_component_bits[mask] = subset_component_bits[prev_mask] | island_component_bits[bit_idx]

        for mask in range(1, num_combinations + 1):
            idx = mask - 1

            boxes[idx, 0] = subset_xmin[mask]
            boxes[idx, 1] = subset_ymin[mask]
            boxes[idx, 2] = subset_xmax[mask]
            boxes[idx, 3] = subset_ymax[mask]
            scores[idx] = subset_area[mask]

            if return_ground_truth:
                used_bits = subset_component_bits[mask]
                if used_bits != 0:
                    first_comp_idx = (used_bits & -used_bits).bit_length() - 1
                    first_source_id = int(source_ids[first_comp_idx])
                    source_full_bits = source_component_bits[first_source_id]
                    # Valid iff proposal-generating components are exactly all components of one source.
                    if used_bits == source_full_bits:
                        if return_instance_targets:
                            gt_instance_bboxes.append(
                                np.array([
                                    subset_xmin[mask],
                                    subset_ymin[mask],
                                    subset_xmax[mask],
                                    subset_ymax[mask],
                                ], dtype=np.float32)
                            )

                            selected_island_labels = []
                            local_mask = mask
                            while local_mask:
                                lsb_local = local_mask & -local_mask
                                local_bit_idx = lsb_local.bit_length() - 1
                                selected_island_labels.append(int(island_ids[local_bit_idx]))
                                local_mask ^= lsb_local

                            if selected_island_labels:
                                instance_mask = np.isin(self.cc_map, selected_island_labels)
                            else:
                                instance_mask = np.zeros_like(self.cc_map, dtype=bool)
                            gt_instance_masks.append(instance_mask.astype(np.uint8))

                            # Class is based on disconnected emission islands in the instance mask.
                            class_label = MCS_LABEL if len(selected_island_labels) > 1 else SCS_LABEL
                            gt_instance_category_ids.append(int(class_label))

                            selected_component_indices = []
                            local_used_bits = used_bits
                            while local_used_bits:
                                lsb_comp = local_used_bits & -local_used_bits
                                selected_component_indices.append(lsb_comp.bit_length() - 1)
                                local_used_bits ^= lsb_comp
                            instance_positions = component_positions[selected_component_indices]
                            gt_instance_positions.append(instance_positions.astype(np.int32, copy=True))

        if return_scores and scores.max() > 0:
            scores /= scores.max()

        if return_ground_truth:
            if return_instance_targets:
                gt_instance_bboxes = np.asarray(gt_instance_bboxes, dtype=np.float32)
                gt_instance_category_ids = np.asarray(gt_instance_category_ids, dtype=np.int32)
                if return_scores:
                    return (
                        boxes,
                        scores,
                        gt_instance_bboxes,
                        gt_instance_masks,
                        gt_instance_category_ids,
                        gt_instance_positions,
                    )
                return (
                    boxes,
                    gt_instance_bboxes,
                    gt_instance_masks,
                    gt_instance_category_ids,
                    gt_instance_positions,
                )

        return (boxes, scores) if return_scores else boxes
