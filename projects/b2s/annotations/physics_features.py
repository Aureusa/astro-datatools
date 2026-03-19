import numpy as np


class PhysicsAwareFeatures:
    def __init__(
            self,
            augmented_proposals: list[np.ndarray],
            valid_positions_list: list[np.ndarray],
            physical_quantities: np.ndarray,
            indeces_to_keep: list[np.ndarray],
            cutout_size: int = 300,
            max_islands: int = 10,
        ):
        """
        Derive physics-aware features for the proposals based on the positions of the components.

        :param augmented_proposals: List of numpy arrays containing proposed bounding boxes for each angle.
        :type augmented_proposals: list[np.ndarray]
        :param valid_positions_list: List of numpy arrays containing valid positions for each angle.
        :type valid_positions_list: list[np.ndarray]
        :param physical_quantities: Array containing physical quantities for each component.
        :type physical_quantities: np.ndarray
        """
        self.augmented_proposals_list = augmented_proposals
        self.valid_positions_list = valid_positions_list
        self.physical_quantities = physical_quantities
        self.indeces_to_keep_list = indeces_to_keep
        self.cutout_size = cutout_size
        self.max_islands = max_islands
        
        # If num_components is < max_islands, we can pad the component
        # positions with dummy values (e.g. (-1, -1)) to ensure that
        # the feature arrays have a consistent shape
        valid_pos_list = []
        for i in range(len(self.valid_positions_list)):
            num_components = len(self.valid_positions_list[i])
            if num_components < max_islands:
                valid_position_arr = self.valid_positions_list[i]
                padding = np.ones((max_islands - valid_position_arr.shape[0], valid_position_arr.shape[1])) * -1
                valid_pos = np.vstack((valid_position_arr, padding))
                valid_pos_list.append(valid_pos)
            else:
                valid_pos_list.append(self.valid_positions_list[i])
        self.valid_positions_list = valid_pos_list
        # Now valid_positions_list is a list of numpy arrays of shape (max_islands, 2) for each angle,
        # where we have padded the arrays with dummy values to keep the same number of components.
        # This is required to ensure that we can train a model that expect fix number of components.

    def derive_features(self, fast: bool = True):
        if fast:
            # TODO: Probably faster but need to check if
            # it is correct and gives the same results as the more detailed method below
            return self._derive_fast(
                self.augmented_proposals_list,
                self.valid_positions_list,
                self.physical_quantities,
                self.indeces_to_keep_list
            )
        scaled_distances, within_proposal_mask = self._derive_geometrical_quantities(
            self.augmented_proposals, self.rotated_all_component_positions
        )

        flux_ratios = self._compute_flux_ratios(
            self.candidates.get("total_flux", []),
            within_proposal_mask
        )

        min_maj_ratios, scaled_majors, scaled_minors = self._compute_maj_min_ratios(
            self.candidates.get("maj", []),
            self.candidates.get("min", []),
            within_proposal_mask
        )

        peak_flux_ratios = self._compute_flux_ratios(
            self.candidates.get("peak_flux", []),
            within_proposal_mask
        )

        # Now all of the features are in the same shape (num_proposals, num_components) and can be used for training
        # We can concatenate them along the feature dimension to get a final feature array of shape (num_proposals, num_components, num_features)
        features = np.stack((scaled_distances, flux_ratios, min_maj_ratios, scaled_majors, scaled_minors, peak_flux_ratios), axis=-1)
        return features, within_proposal_mask

    def derive_ground_truth(self, source_components_arr: np.ndarray, within_proposal_mask):
        """
        Generate per-proposal component membership and proposal validity targets.

        :param source_components_arr: Source/component names with shape (2, num_components_original).
        :type source_components_arr: np.ndarray
        :param within_proposal_mask: Proposal-component mask either as a single array
            (num_proposals, num_components) or list of such arrays (per angle).
        :type within_proposal_mask: np.ndarray | list[np.ndarray]
        :return: Ground-truth component membership and proposal validity.
            Returns arrays for homogeneous case, lists for per-angle inhomogeneous case.
        :rtype: tuple[np.ndarray, np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]
        """
        if source_components_arr.shape[0] != 2:
            raise ValueError(
                f"source_components_arr must have shape (2, num_components), got {source_components_arr.shape}."
            )

        if isinstance(within_proposal_mask, list):
            gt_membership_list = []
            gt_validity_list = []
            for angle_idx, angle_mask in enumerate(within_proposal_mask):
                angle_arr = self._build_angle_source_components_arr(
                    source_components_arr,
                    self.indeces_to_keep_list[angle_idx],
                    angle_mask,
                )
                gt_membership, gt_validity = self._generate_gt_component_membership(
                    angle_arr,
                    angle_mask,
                )
                gt_membership_list.append(gt_membership)
                gt_validity_list.append(gt_validity)
            return gt_membership_list, gt_validity_list

        angle_arr = self._build_angle_source_components_arr(
            source_components_arr,
            self.indeces_to_keep_list[0],
            within_proposal_mask,
        )
        return self._generate_gt_component_membership(angle_arr, within_proposal_mask)

    @staticmethod
    def _mask_num_components(mask: np.ndarray, min_components: int) -> int:
        """Infer component-axis size for either (P, C) or (C, P) mask layout."""
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"within_proposal_mask must be 2D, got shape {mask.shape}.")
        dims = [int(mask.shape[0]), int(mask.shape[1])]
        valid_dims = [d for d in dims if d >= min_components]
        if valid_dims:
            return min(valid_dims)
        return max(dims)

    @staticmethod
    def _build_angle_source_components_arr(
        source_components_arr: np.ndarray,
        kept_component_indices: np.ndarray,
        within_proposal_mask: np.ndarray,
    ) -> np.ndarray:
        """Slice source/component names for one angle and pad metadata to mask component width."""
        source_names = np.asarray(source_components_arr[0])[kept_component_indices]
        component_names = np.asarray(source_components_arr[1])[kept_component_indices]

        target_components = PhysicsAwareFeatures._mask_num_components(
            within_proposal_mask,
            min_components=int(np.asarray(kept_component_indices).shape[0]),
        )
        curr_components = source_names.shape[0]

        if curr_components < target_components:
            pad_count = target_components - curr_components
            source_pad = np.array([f"__PAD_SOURCE_{i}" for i in range(pad_count)], dtype=object)
            component_pad = np.array([f"__PAD_COMPONENT_{i}" for i in range(pad_count)], dtype=object)
            source_names = np.concatenate((source_names, source_pad), axis=0)
            component_names = np.concatenate((component_names, component_pad), axis=0)
        elif curr_components > target_components:
            source_names = source_names[:target_components]
            component_names = component_names[:target_components]

        return np.vstack((source_names, component_names))

    @staticmethod
    def _generate_gt_component_membership(source_components_arr: np.ndarray, within_proposal_mask: np.ndarray):
        """Vectorized strict GT generation: exactly one complete source per valid proposal."""
        mask = np.asarray(within_proposal_mask, dtype=bool)
        num_components = source_components_arr.shape[1]

        if mask.shape[-1] == num_components:
            # Already (num_proposals, num_components)
            pass
        elif mask.shape[0] == num_components:
            # Convert from (num_components, num_proposals)
            mask = mask.T
        else:
            raise ValueError(
                "within_proposal_mask must align with num_components in source_components_arr. "
                f"Got mask shape {mask.shape} and num_components={num_components}."
            )

        sources = np.asarray(source_components_arr[0])
        _, source_ids = np.unique(sources, return_inverse=True)
        num_sources = int(source_ids.max()) + 1

        source_component_matrix = np.zeros((num_sources, num_components), dtype=np.int32)
        source_component_matrix[source_ids, np.arange(num_components)] = 1

        counts_per_source = mask.astype(np.int32) @ source_component_matrix.T
        source_sizes = source_component_matrix.sum(axis=1)
        total_components_per_proposal = mask.sum(axis=1)

        contains_all_of_source = counts_per_source == source_sizes[np.newaxis, :]
        contains_only_that_source = total_components_per_proposal[:, np.newaxis] == source_sizes[np.newaxis, :]
        valid_source_per_proposal = contains_all_of_source & contains_only_that_source

        gt_component_membership = (
            valid_source_per_proposal.astype(np.int32) @ source_component_matrix
        ) > 0
        gt_component_membership = gt_component_membership.astype(np.int32)
        gt_proposal_validity = np.any(valid_source_per_proposal, axis=1).astype(np.int32)
        return gt_component_membership, gt_proposal_validity

    def _derive_fast(self, augmented_proposals, valid_positions, physical_quantities, indeces_to_keep):
        try:
            proposals = np.array(augmented_proposals) # shape (num_proposals, 4)
        except ValueError:
            # This can happen if the array has an inhomogeneous shape, which can
            # occur if there are different number of proposals for some angles.
            # This happens because some components go out of the cutout after rotation and croping,
            # which results in different number of valid positions and thus different number of proposals
            # after segmentation and proposal generation.
            # In this case we can recursively call this function for each angle separately and concatenate the results
            features_list = []
            within_proposal_mask_list = []
            for i in range(len(augmented_proposals)): # Loop over all angles
                features, within_proposal_mask = self._derive_fast(
                    augmented_proposals[i], valid_positions[i], physical_quantities, indeces_to_keep[i]
                )
                features_list.append(features)
                within_proposal_mask_list.append(within_proposal_mask)
            return features_list, within_proposal_mask_list
        
        if len(proposals.shape) == 3: # (angles, num_proposals, 4)
            # Just get the first angle as the features are the same for all angles,
            # since they are derived from the same component positions and proposals
            proposals = proposals[0] # shape (num_proposals, 4)
            valid_positions = valid_positions[0] # shape (num_components, 2)
            indeces_to_keep = indeces_to_keep[0] # shape (num_components,)
            physical_quantities = physical_quantities[:, indeces_to_keep] # shape (num_physical_quantities, num_components)
        else:
            physical_quantities = physical_quantities[:, indeces_to_keep] # shape (num_physical_quantities, num_components)

        # If num_components in physical_quantities is less than max_islands,
        # we can pad the physical quantities with zeros to ensure that they have a consistent shape!
        # We assume that the physical quantities are structured in this order:
        # total_flux, peak_flux, maj, min
        if physical_quantities.shape[1] < valid_positions.shape[0]:
            padding = np.zeros((physical_quantities.shape[0], valid_positions.shape[0] - physical_quantities.shape[1]))
            physical_quantities = np.hstack((physical_quantities, padding)) # shape (num_physical_quantities, num_components)
     
        scaled_distances, scaled_dx, scaled_dy, sin_theta, cos_theta, within_proposal_mask = self._derive_geometrical_quantities(
            proposals, valid_positions
        ) # shape (num_proposals, num_components); mask (num_components, num_proposals)

        # TODO: SOMETHING GOES WRONG HERE \/ \/ \/ \/
        # Add another axis for proposals and mask out the components that are not within each proposal
        # shape (num_proposals, num_components, num_features)
        proposal_features = np.where(
            within_proposal_mask.T[:, :, np.newaxis],   # (31, 10, 1)
            physical_quantities.T[np.newaxis, :, :],    # (1, 10, 4)
            0
        )   # result: (31, 10, 4)
        # /\ /\ /\ /\ /\ /\ /\ 

        # Find the maximum value of each feature within each proposal to get a single feature vector per proposal
        # shape (num_proposals, num_features)
        max_features = np.max(proposal_features, axis=1)

        # Compute the minor-to-major axis ratio for each proposal using the max major and minor axis lengths within each proposal
        # shape (num_proposals, num_components)
        min_maj_ratios = (proposal_features[:, :, 3] / np.where(proposal_features[:, :, 2] == 0, 1, proposal_features[:, :, 2]))

        # Scale the features to the range [0, 1] by dividing SAFELY by the maximum value of each feature across all proposals
        # shape (num_proposals, num_components, num_features)
        scaled_features = proposal_features / np.where(max_features == 0, 1, max_features)[:, np.newaxis, :]

        # Add the minmaj ratio and the scaled distances to the feature dimension
        features = np.concatenate((
            scaled_features,
            min_maj_ratios[:, :, np.newaxis],
            scaled_distances[:, :, np.newaxis],
            scaled_dx[:, :, np.newaxis],
            scaled_dy[:, :, np.newaxis],
            sin_theta[:, :, np.newaxis],
            cos_theta[:, :, np.newaxis],
        ), axis=-1)
        return (
            features,   # shape (num_proposals, num_components, num_features)
            within_proposal_mask.T # shape (num_proposals, num_components)
        )
    
    def _compute_flux_ratios(self, fluxes: list[float], within_proposal_mask: np.ndarray):
        """
        Compute flux ratios for components within each proposal.

        :param fluxes: List of flux values for all components.
        :type fluxes: list[float]
        :param within_proposal_mask: Boolean mask indicating which components are within each proposal.
        :type within_proposal_mask: np.ndarray
        :return: Array of flux ratios for each proposal.
        :rtype: np.ndarray
        """
        # We get the proposal fluxes which correspond to the fluxes within each proposal box
        # This has a shape of (num_proposals, num_components) - total number of components
        # Some components may be outside the proposal box, in which case we set their flux to 0 for that proposal
        proposal_fluxes = np.where(within_proposal_mask, fluxes[:, np.newaxis], 0)

        flux_ratios = self._scale_to_0_1(proposal_fluxes)
        return flux_ratios.reshape((proposal_fluxes.shape[-1], proposal_fluxes.shape[0]))  # shape (num_proposals, num_components)

    def _compute_maj_min_ratios(self, majors: list[float], minors: list[float], within_proposal_mask: np.ndarray):
        """
        Compute major-to-minor axis ratios for components within each proposal.

        :param majors: List of major axis lengths for all components.
        :type majors: list[float]
        :param minors: List of minor axis lengths for all components.
        :type minors: list[float]
        :param within_proposal_mask: Boolean mask indicating which components are within each proposal.
        :type within_proposal_mask: np.ndarray
        :return: Array of major-to-minor axis ratios for each proposal.
        :rtype: np.ndarray
        """
        # Similar to flux ratios, we compute the major-to-minor axis ratio for components within each proposal
        proposal_majors = np.where(within_proposal_mask, majors[:, np.newaxis], 0)
        proposal_minors = np.where(within_proposal_mask, minors[:, np.newaxis], 0)

        # We compute the minor-to-major axis ratio for each component within each proposal,
        # which gives us a measure of how elongated the component is within that proposal
        # Also it is in a nice scale between 0 and 1, where 1 corresponds to a perfectly circular
        # component and values close to 0 correspond to very elongated components

        # this means that some components are outside the proposal box,
        # we set their major axis to 1 to avoid division by zero and get a minor-to-major ratio of 0 for those components
        proposal_majors[proposal_majors == 0] = 1

        # shape (num_proposals, num_components)
        min_maj_ratios = (proposal_minors / proposal_majors).reshape((proposal_majors.shape[-1], proposal_majors.shape[0]))
        
        # Also we can scale the major and minor axis by their corresponding maximum within each proposal
        # to get a relative size of the component within that proposal
        # shape (num_proposals, num_components)
        scaled_majors = self._scale_to_0_1(proposal_majors).reshape((proposal_majors.shape[-1], proposal_majors.shape[0]))
        scaled_minors = self._scale_to_0_1(proposal_minors).reshape((proposal_minors.shape[-1], proposal_minors.shape[0]))
        return min_maj_ratios, scaled_majors, scaled_minors

    def _scale_to_0_1(self, values: np.ndarray):
        """
        Scale an array of values to the range [0, 1].

        :param values: Array of values to scale.
        :type values: np.ndarray
        :return: Scaled array of values in the range [0, 1].
        :rtype: np.ndarray
        """
        max_values = np.max(values, axis=0)
        max_values[max_values == 0] = 1
        return values / max_values[np.newaxis, :]

    def _derive_geometrical_quantities(self, proposals: np.ndarray, component_positions_array: np.ndarray):
        """
        Derive geometrical quantities (e.g. relative distance to the center of the proposals of each component)

        :param rotated_all_component_positions: List of lists of (x, y) positions for all components for each angle.
        :type rotated_all_component_positions: list[list[tuple[int, int]]]
        :param augmented_proposals: List of numpy arrays containing proposed bounding boxes for each angle.
        :type augmented_proposals: list[np.ndarray]
        """        
        proposal_centers = np.column_stack((
            (proposals[:, 0] + proposals[:, 2]) / 2,  # x_center
            (proposals[:, 1] + proposals[:, 3]) / 2   # y_center
        ))  # shape (num_proposals, 2)

        # Compute distances from each component to each proposal center
        # We can use broadcasting to compute this efficiently
        distances = np.linalg.norm(
            component_positions_array[:, np.newaxis, :] - proposal_centers[np.newaxis, :, :],
            axis=-1
        )  # shape (num_components, num_proposals)

        # Now we compute dx = x_component - x_proposal_center and dy = y_component - y_proposal_center
        dx = component_positions_array[:, np.newaxis, 0] - proposal_centers[np.newaxis, :, 0]  # shape (num_components, num_proposals)
        dy = component_positions_array[:, np.newaxis, 1] - proposal_centers[np.newaxis, :, 1]  # shape (num_components, num_proposals)

        # We keep only those that are within the proposal bounding box (i.e. where dx < width/2 and dy < height/2)
        proposal_widths = proposals[:, 2] - proposals[:, 0]  # shape (num_proposals,)
        proposal_heights = proposals[:, 3] - proposals[:, 1]  # shape (num_proposals,)

        within_proposal_mask = (
            (np.abs(dx) < proposal_widths[np.newaxis, :] / 2) &
            (np.abs(dy) < proposal_heights[np.newaxis, :] / 2)
        )  # shape (num_components, num_proposals)

        # Keep only distances for components inside each proposal; outside values are 0.
        valid_distances = np.where(within_proposal_mask, distances, 0.0)

        # Scale to cutout size to keep distances in a comparable range.
        # shape (num_proposals, num_components)
        scaled_distances = (valid_distances / self.cutout_size).T

        # Use safe widths/heights to avoid division by zero for degenerate proposals.
        safe_proposal_widths = np.where(proposal_widths == 0, 1.0, proposal_widths)
        safe_proposal_heights = np.where(proposal_heights == 0, 1.0, proposal_heights)

        # Scale dx and dy by proposal size to get relative positions within each proposal box.
        scaled_dx_raw = np.where(within_proposal_mask, dx / safe_proposal_widths[np.newaxis, :], 0.0)
        scaled_dy_raw = np.where(within_proposal_mask, dy / safe_proposal_heights[np.newaxis, :], 0.0)
        scaled_dx = scaled_dx_raw.T  # shape (num_proposals, num_components)
        scaled_dy = scaled_dy_raw.T  # shape (num_proposals, num_components)

        # Directional encoding in normalized proposal coordinates.
        # sin/cos avoids angle discontinuity at +/-pi and is easier for MLPs to learn.
        theta = np.arctan2(scaled_dy_raw, scaled_dx_raw)
        sin_theta = np.where(within_proposal_mask, np.sin(theta), 0.0).T  # shape (num_proposals, num_components)
        cos_theta = np.where(within_proposal_mask, np.cos(theta), 0.0).T  # shape (num_proposals, num_components)
        return scaled_distances, scaled_dx, scaled_dy, sin_theta, cos_theta, within_proposal_mask
    