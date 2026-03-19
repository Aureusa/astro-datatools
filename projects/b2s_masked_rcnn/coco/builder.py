import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from astropy.coordinates import SkyCoord
import astropy.units as u

from astro_datatools.logger import setup_logging
from astro_datatools.core.datasets.coco.builder import CocoDatasetBuilderBase

from .sample import LoTSS_B2S_MaskRCNN_Sample
from .category import LoTSS_B2S_SCS_CocoCategory, LoTSS_B2S_MCS_CocoCategory

from b2s_masked_rcnn.annotations import annotate_and_augment


B2S_SCS_CATEGORY = LoTSS_B2S_SCS_CocoCategory(id=1, name="SCS")
B2S_MCS_CATEGORY = LoTSS_B2S_MCS_CocoCategory(id=2, name="MCS")


class B2SDatasetBuilder(CocoDatasetBuilderBase):
    def __init__(
            self,
            cutouts: list,
            component_catalogue: pd.DataFrame,
            rotation_angles: list[int],
            crop_size: int,
            max_precomputed_islands: int,
            nr_sigmas: int,
            rms: float,
            stretch_type: str,
            segmentation_mode: str,
            class_ratio: float,
            workers: int,
            save_dir: str,
        ):
        self.logger = setup_logging(name=f"b2s_masked_rcnn.coco.{self.__class__.__name__}")
        self.cutouts = cutouts
        self.component_catalogue = component_catalogue

        if stretch_type not in ["sqrt_stretch", "asinh_stretch"]:
            self.logger.error(f"Invalid stretch type: {stretch_type}. Must be 'sqrt_stretch' or 'asinh_stretch'.")
            raise ValueError(f"Invalid stretch type: {stretch_type}. Must be 'sqrt_stretch' or 'asinh_stretch'.")

        self.segmentation_mode = segmentation_mode

        # Make sure save directory exists
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.rotation_angles = rotation_angles
        self.crop_size = crop_size
        self.max_precomputed_islands = max_precomputed_islands
        self.rms = rms
        self.nr_sigmas = nr_sigmas
        self.stretch_type = stretch_type
        self.class_ratio = class_ratio
        self.workers = workers

        if "RA" not in component_catalogue.columns or "DEC" not in component_catalogue.columns:
            raise ValueError("component_catalogue must contain 'RA' and 'DEC' columns.")

        source_name_col = None
        for candidate in ["Parent_Source", "Source_Name", "Component_Name"]:
            if candidate in component_catalogue.columns:
                source_name_col = candidate
                break
        if source_name_col is None:
            self.logger.error(
                "component_catalogue must contain one of: 'Parent_Source', 'Source_Name', 'Component_Name'."
            )
            raise ValueError(
                "component_catalogue must contain one of: 'Parent_Source', 'Source_Name', 'Component_Name'."
            )

        component_name_col = None
        for candidate in ["Component_Name", "Parent_Source", "Source_Name"]:
            if candidate in component_catalogue.columns:
                component_name_col = candidate
                break
        if component_name_col is None:
            self.logger.error(
                "component_catalogue must contain one of: 'Component_Name', 'Parent_Source', 'Source_Name'."
            )
            raise ValueError(
                "component_catalogue must contain one of: 'Component_Name', 'Parent_Source', 'Source_Name'."
            )

        catalog_ra = np.asarray(component_catalogue["RA"].to_numpy(), dtype=np.float64)
        catalog_dec = np.asarray(component_catalogue["DEC"].to_numpy(), dtype=np.float64)
        raw_source_names = component_catalogue[source_name_col].to_numpy()
        raw_component_names = component_catalogue[component_name_col].to_numpy()

        catalog_source_names = np.array([
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in raw_source_names
        ], dtype=object)
        catalog_component_names = np.array([
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in raw_component_names
        ], dtype=object)

        sort_indices = np.argsort(catalog_ra, kind="stable")
        self._catalog_sort_idx = sort_indices
        self._catalog_ra_sorted = catalog_ra[sort_indices]
        self._catalog_dec_sorted = catalog_dec[sort_indices]
        self._catalog_source_names_sorted = catalog_source_names[sort_indices]
        self._catalog_component_names_sorted = catalog_component_names[sort_indices]

        # TODO: This is not clean way as it hardcodes the following column names and assumes they exist.
        self._catalog_total_flux_sorted = component_catalogue["Total_flux"].to_numpy()[sort_indices]
        self._catalog_peak_flux_sorted = component_catalogue["Peak_flux"].to_numpy()[sort_indices]
        self._catalog_maj_sorted = component_catalogue["Maj"].to_numpy()[sort_indices]
        self._catalog_min_sorted = component_catalogue["Min"].to_numpy()[sort_indices]

        # Metrics:
        self.no_proposal_generated_count = 0
        self.no_components_found = 0

    def build(self) -> dict:
        """
        Build the COCO dataset by generating samples, categories, and saving to a JSON file.

        :return: Dictionary representing the COCO dataset.
        :rtype: dict
        """
        coco = super().build()

        if self.no_proposal_generated_count > 0:
            unique_objects_skipped = self.no_proposal_generated_count / len(self.rotation_angles)
            self.logger.info(
                f"Skipped {self.no_proposal_generated_count} cutouts where no proposals were generated. "
                f"This means that {unique_objects_skipped} unique objects were skipped."
            )
        if self.no_components_found > 0:
            self.logger.info(f"Skipped {self.no_components_found} cutouts where no components were found in the catalog.")
        return coco

    def _get_filepath(self) -> str:
        return os.path.join(self.save_dir, "annotations.json")
        
    def _add_categories(self, coco: dict) -> dict:
        coco['categories'].append(B2S_SCS_CATEGORY.to_dict())
        coco['categories'].append(B2S_MCS_CATEGORY.to_dict())
        return coco

    def _register_sample(self, sample: LoTSS_B2S_MaskRCNN_Sample, coco: dict) -> dict:
        """Register one image plus zero or more instance annotations."""
        result = sample.register_sample()
        if result is None:
            return coco

        coco["images"].append(result["image"])

        annotations = result.get("annotations", [])
        for ann in annotations:
            ann["id"] = self.next_annotation_id
            self.next_annotation_id += 1
            coco["annotations"].append(ann)
        return coco

    def _process_single_cutout_thread(self, cutout, cutout_index, coco, coco_lock):
        """
        Process a single cutout in a thread and register samples immediately.
        IDs are deterministically calculated: cutout_index * num_rotations + angle_index + 1
        """
        try:
            generated_cutouts = self._generate_positive_samples(
                cutout,
                coco,
                coco_lock
            )
        except Exception as e:
            self.logger.error(f"Error in thread processing cutout (idx {cutout_index}): {e}", exc_info=True)

    def _generate_positive_samples(self, cutout, coco, coco_lock) -> int:
        data = None
        rgb_rotated_data = None
        rotated_all_component_positions = None
        rotated_proposed_boxes = None
        rotated_proposal_scores = None

        try:
            # Get the data and the positions from the cutout
            data = cutout.get_data()
            candidates = self._get_cutout_candidates(cutout)
            if candidates is None:
                self.logger.debug(
                    f"Skipping cutout at RA: {cutout.ra}, DEC: {cutout.dec} - no candidate components found."
                )
                self.no_components_found += 1
                return 0
                
            # Annotate and augment the data
            (
                rgb_rotated_data,
                rotated_all_component_positions,
                gt_instance_bboxes,
                gt_instance_masks,
                gt_instance_category_ids,
                rotated_proposed_boxes,
                rotated_proposal_scores,
                grouping_metadata,
            ) = annotate_and_augment(
                data=data,
                candidates=candidates,
                angles=self.rotation_angles,
                specific_crop_size=(self.crop_size, self.crop_size),
                dynamic_cropping=False,
                max_precomputed_islands=self.max_precomputed_islands,
                nr_sigmas=self.nr_sigmas,
                rms=self.rms,
                asinh_stretch=False if self.stretch_type == "sqrt_stretch" else True
            )

            # Create and register samples for all rotations
            origin_id = None  # Will be set to the ID of the first valid sample
            generated_cutouts = 0
            for angle_index, angle in enumerate(self.rotation_angles):
                curr_proposed_boxes = rotated_proposed_boxes[angle_index]

                # Keep image/annotation/proposal files in sync: skip angles with no proposals.
                if curr_proposed_boxes is None or len(curr_proposed_boxes) == 0:
                    self.logger.warning(
                        f"Skipping cutout RA={cutout.ra}, DEC={cutout.dec}, angle={angle}: no proposals generated."
                    )
                    self.no_proposal_generated_count += 1
                    continue

                curr_gt_instance_bboxes = (
                    gt_instance_bboxes[angle_index]
                    if isinstance(gt_instance_bboxes, list)
                    else gt_instance_bboxes
                )
                curr_gt_instance_masks = (
                    gt_instance_masks[angle_index]
                    if isinstance(gt_instance_masks, list)
                    else gt_instance_masks
                )
                curr_gt_instance_category_ids = (
                    gt_instance_category_ids[angle_index]
                    if isinstance(gt_instance_category_ids, list)
                    else gt_instance_category_ids
                )
                curr_proposal_scores = (
                    rotated_proposal_scores[angle_index]
                    if isinstance(rotated_proposal_scores, list)
                    else rotated_proposal_scores
                )

                # Get next sequential ID (thread-safe)
                with coco_lock:
                    sample_id = self.next_id
                    self.next_id += 1
                    
                    # Set origin_id to the first valid sample ID for this cutout
                    if origin_id is None:
                        origin_id = sample_id

                candidates_for_angle = {}
                if grouping_metadata is not None and "angles" in grouping_metadata:
                    candidates_for_angle["grouping"] = grouping_metadata["angles"][angle_index]
                sample = LoTSS_B2S_MaskRCNN_Sample(
                    id=sample_id,
                    image_id=sample_id,
                    ra=cutout.ra,
                    dec=cutout.dec,
                    rgb_image=rgb_rotated_data[angle_index],
                    proposed_boxes=curr_proposed_boxes,
                    proposal_scores=curr_proposal_scores,
                    instance_bboxes=curr_gt_instance_bboxes,
                    instance_masks=curr_gt_instance_masks,
                    instance_category_ids=curr_gt_instance_category_ids,
                    candidates=candidates_for_angle,
                    rotated=True if angle != 0 else False,
                    origin_id=origin_id,
                    rotation_angle=angle,
                    stretch=self.stretch_type,
                    reprojected=False,
                    old_redshift=None,
                    new_redshift=None,
                    directory=self.save_dir,
                    save_image=True
                )
                
                # Register sample with thread-safe lock
                with coco_lock:
                    coco = self._register_sample(sample, coco)

                generated_cutouts += 1

            return generated_cutouts
        finally:
            # Free memory
            del data
            del rgb_rotated_data
            del rotated_proposed_boxes
            del rotated_proposal_scores
            del rotated_all_component_positions
            gc.collect()

    def _populate_samples(self, coco: dict) -> dict:
        # Pre-create directories to avoid repeated existence checks
        images_dir = os.path.join(self.save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Thread-safe lock for COCO dict updates and ID counter
        coco_lock = threading.Lock()
        
        # Shared counter for sequential IDs (no gaps)
        self.next_id = 1
        self.next_annotation_id = 1
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all cutout processing tasks
            future_to_index = {}
            for cutout_index, cutout in enumerate(self.cutouts):
                future = executor.submit(
                    self._process_single_cutout_thread,
                    cutout,
                    cutout_index,
                    coco,
                    coco_lock
                )
                future_to_index[future] = cutout_index
            
            # Wait for all to complete with progress bar
            with tqdm(total=len(self.cutouts), desc=f"Generating LoTSS Samples for COCO Dataset") as pbar:
                for future in as_completed(future_to_index):
                    cutout_index = future_to_index[future]
                    try:
                        future.result()  # Samples already registered inside thread
                    except Exception as e:
                        self.logger.error(f"Error processing cutout {cutout_index}: {e}", exc_info=True)
                    finally:
                        pbar.update(1)
                        
        return coco

    def _get_cutout_candidates(self, curr_cutout):
        max_sep_arcsec = curr_cutout.size_arcmin * 60 / 2
        cos_dec = np.cos(np.deg2rad(curr_cutout.dec))
        if abs(cos_dec) < 1e-8:
            cos_dec = 1e-8

        delta_ra = max_sep_arcsec / cos_dec / 3600
        delta_dec = max_sep_arcsec / 3600

        ra_min = curr_cutout.ra - delta_ra
        ra_max = curr_cutout.ra + delta_ra
        dec_min = curr_cutout.dec - delta_dec
        dec_max = curr_cutout.dec + delta_dec

        left_idx = np.searchsorted(self._catalog_ra_sorted, ra_min, side="left")
        right_idx = np.searchsorted(self._catalog_ra_sorted, ra_max, side="right")

        if left_idx >= right_idx:
            return None

        candidate_dec = self._catalog_dec_sorted[left_idx:right_idx]
        dec_mask = (candidate_dec >= dec_min) & (candidate_dec <= dec_max)
        if not np.any(dec_mask):
            return None

        candidate_ra = self._catalog_ra_sorted[left_idx:right_idx][dec_mask]
        candidate_dec = candidate_dec[dec_mask]
        candidate_source_names = self._catalog_source_names_sorted[left_idx:right_idx][dec_mask]
        candidate_component_names = self._catalog_component_names_sorted[left_idx:right_idx][dec_mask]
        candidate_orig_idx = self._catalog_sort_idx[left_idx:right_idx][dec_mask]

        coords = SkyCoord(ra=candidate_ra * u.deg, dec=candidate_dec * u.deg, frame="icrs")
        x_pixels, y_pixels = curr_cutout.get_wcs().world_to_pixel(coords)

        x_pixels = np.rint(x_pixels).astype(np.int32)
        y_pixels = np.rint(y_pixels).astype(np.int32)

        valid_mask = (
            (x_pixels >= 0)
            & (x_pixels < curr_cutout.size_pixels)
            & (y_pixels >= 0)
            & (y_pixels < curr_cutout.size_pixels)
        )
        if not np.any(valid_mask):
            return None

        source_names = candidate_source_names[valid_mask]
        component_names = candidate_component_names[valid_mask]
        x_pixels = x_pixels[valid_mask]
        y_pixels = y_pixels[valid_mask]
        candidate_orig_idx = candidate_orig_idx[valid_mask]

        # Physical quantities
        total_flux = self._catalog_total_flux_sorted[left_idx:right_idx][dec_mask][valid_mask]
        peak_flux = self._catalog_peak_flux_sorted[left_idx:right_idx][dec_mask][valid_mask]
        maj = self._catalog_maj_sorted[left_idx:right_idx][dec_mask][valid_mask]
        min = self._catalog_min_sorted[left_idx:right_idx][dec_mask][valid_mask]

        # Preserve original catalogue row ordering for deterministic behavior.
        order = np.argsort(candidate_orig_idx, kind="stable")
        return {
            "source_names": source_names[order],
            "component_names": component_names[order],
            "xy_list": list(zip(x_pixels[order], y_pixels[order])),
            "total_flux": total_flux[order],
            "peak_flux": peak_flux[order],
            "maj": maj[order],
            "min": min[order],
        }
