import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append('/home/penchev/astro-datatools/')
from astro_datatools.lotss_annotations import Segment
from astro_datatools.core.datasets.coco.builder import CocoDatasetBuilderBase

from .sample import LoTSS_GRG_Sample, LoTSS_Search_Sample, LoTSS_Negative_GRG_Sample
from .category import LoTSS_GRG_CocoCategory
from .clean import COCODatasetCleaner
from .evaluator import GTEvaluator

# Append project path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from grg_detection.annotations import annotate_and_augment, augment_and_get_proposals, annotate, GRGFinder

sys.path.append('/home/penchev/strw_lofar_data_utils/')
from src.core.cutout_maker.cutout_catalogue import CutoutCatalogue
from src.core.cutout_maker.make_cutout import make_cutout


logger = logging.getLogger("GRGDatasetBuilder")


GRG_CATEGORY = LoTSS_GRG_CocoCategory(id=1, name="GRG")


class GRGDatasetBuilder(CocoDatasetBuilderBase):
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
            save_dir: str
        ):
        self.cutouts = cutouts
        self.component_catalogue = component_catalogue

        if stretch_type not in ["sqrt_stretch", "asinh_stretch"]:
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

    def build(self) -> dict:
        """
        Build the COCO dataset by generating samples, categories, and saving to a JSON file.

        :return: Dictionary representing the COCO dataset.
        :rtype: dict
        """
        coco = super().build()

        # Clean the dataset using COCODatasetCleaner
        logger.info("Cleaning the COCO dataset...")
        filepath_for_coco = self._get_filepath()
        cleaner = COCODatasetCleaner(
            coco, filepath_for_coco, filepath_for_coco
        )
        coco, _, updated_images_ids, removed_images_ids = cleaner.clean(
            save_cleaned_dataset=True
        )
        logger.info(f"Removed {len(removed_images_ids)} images without annotations.")
        logger.info(f"Updated {len(updated_images_ids)} images with new annotations.")

        # Evaluate against ground truth, should get perfect scores after cleaning
        logger.info("Evaluating the COCO dataset against ground truth...")
        gt_evaluator = GTEvaluator(coco, filepath_for_coco)
        results = gt_evaluator.evaluate()
        if isinstance(results, dict):
            info = "Results of evaluation against ground truth:\n"
            for key, value in results.items():
                info += f"{key}: {value}\n"
            logger.info(info)

        return coco

    def _get_filepath(self) -> str:
        return os.path.join(self.save_dir, "annotations.json")
        
    def _add_categories(self, coco: dict) -> dict:
        coco['categories'].append(GRG_CATEGORY.to_dict())
        return coco

    def _process_single_cutout_thread(self, cutout, cutout_index, coco, coco_lock):
        """
        Process a single cutout in a thread and register samples immediately.
        IDs are deterministically calculated: cutout_index * num_rotations + angle_index + 1
        """
        try:
            generated_cutouts = self._generate_positive_samples(
                cutout,
                cutout_index,
                coco,
                coco_lock
            )

            num_negative_examples = int(generated_cutouts * self.class_ratio)
            if num_negative_examples > 0:
                self._generate_negative_samples(
                    cutout,
                    cutout_index,
                    coco,
                    coco_lock,
                    num_negative_examples
                )
            
        except Exception as e:
            logger.error(f"Error in thread processing cutout {cutout_index}: {e}", exc_info=True)

    def _generate_positive_samples(self, cutout, cutout_index, coco, coco_lock) -> int:
        data = None
        grg_positions = None
        all_component_positions = None
        rgb_rotated_data = None
        rotated_grg_positions = None
        rotated_all_component_positions = None
        rotated_grg_segs = None
        rotated_grg_bboxes = None
        rotated_proposed_boxes = None
        proposal_scores = None

        try:
            # Get the data and the positions from the cutout
            data = cutout.get_data()
            grg_positions, all_component_positions = self._get_positions(
                cutout,
                data,
                nr_sigmas=self.nr_sigmas,
                rms=self.rms
            )
            
            if grg_positions is False:
                logger.warning(f"Skipping cutout at RA: {cutout.ra}, DEC: {cutout.dec} - no GRG annotation found.")
                return 0
                
            # Annotate and augment the data
            (
                rgb_rotated_data,
                rotated_grg_positions,
                rotated_all_component_positions,
                rotated_grg_segs,
                rotated_grg_bboxes,
                rotated_proposed_boxes,
                proposal_scores
            ) = annotate_and_augment(
                data=data,
                grg_positions=grg_positions,
                all_component_positions=all_component_positions,
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
                # Skip this rotation if bbox is None (no valid segmentation)
                if rotated_grg_bboxes[angle_index] is None:
                    logger.debug(f"Skipping rotation {angle}° for cutout {cutout_index} - no valid bbox")
                    continue
                
                # Get next sequential ID (thread-safe)
                with coco_lock:
                    sample_id = self.next_id
                    self.next_id += 1
                    
                    # Set origin_id to the first valid sample ID for this cutout
                    if origin_id is None:
                        origin_id = sample_id
                
                sample = LoTSS_GRG_Sample(
                    id=sample_id,
                    image_id=sample_id,
                    category_id=1,
                    ra=cutout.ra,
                    dec=cutout.dec,
                    rgb_image=rgb_rotated_data[angle_index],
                    full_annotation=(rotated_grg_segs[angle_index], rotated_grg_bboxes[angle_index]),
                    proposed_boxes=rotated_proposed_boxes[angle_index],
                    proposal_scores=proposal_scores[angle_index],
                    grg_positions=rotated_grg_positions[angle_index],
                    all_component_positions=rotated_all_component_positions[angle_index],
                    rotated=True if angle != 0 else False,
                    origin_id=origin_id,
                    rotation_angle=angle,
                    stretch=self.stretch_type,
                    segmentation_mode=self.segmentation_mode,
                    reprojected=False,
                    old_redshift=None,
                    new_redshift=None,
                    iscrowd=0,
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
            del grg_positions
            del all_component_positions
            del rgb_rotated_data
            del rotated_grg_segs
            del rotated_grg_bboxes
            del rotated_proposed_boxes
            del rotated_grg_positions
            del rotated_all_component_positions
            gc.collect()

    def _generate_negative_samples(
            self,
            cutout,
            cutout_index,
            coco,
            coco_lock,
            num_negative_examples
        ) -> None:
        mosaic = cutout.mosaic
        valid_data_radius_deg = mosaic.valid_data_radius_deg
        valid_data_radius_deg_squared = valid_data_radius_deg ** 2
        mosaic_center_ra = mosaic.ra
        mosaic_center_dec = mosaic.dec

        # Generate cutouts with random RA/DEC pairs within the valid data radius
        registered_samples = 0
        while registered_samples < num_negative_examples:
            # Generate random RA/DEC offsets within the valid data radius
            ra_offset = np.random.uniform(-valid_data_radius_deg, valid_data_radius_deg)
            dec_offset = np.random.uniform(-valid_data_radius_deg, valid_data_radius_deg)
            
            # Calculate new RA/DEC
            new_ra = mosaic_center_ra + ra_offset
            new_dec = mosaic_center_dec + dec_offset
            
            # Check if the new position is within the valid data radius
            if ra_offset**2 + dec_offset**2 <= valid_data_radius_deg_squared:
                cutout = make_cutout(
                    mosaic,
                    new_ra,
                    new_dec,
                    size_arcmin=None,
                    size_pixels=self.crop_size
                )
                if cutout is not None:
                    # Get the data and the positions from the cutout
                    data = cutout.get_data()
                    positions = self._get_positions_negative_grg(
                        cutout,
                        nr_sigmas=self.nr_sigmas,
                        rms=self.rms
                    )
                    
                    if len(positions) == 0:
                        continue
                        
                    # Annotate and augment the data
                    (
                        augmented_data,
                        proposed_boxes,
                        proposal_scores,
                    ) = augment_and_get_proposals(
                        data=data,
                        positions=positions,
                        max_precomputed_islands=self.max_precomputed_islands,
                        nr_sigmas=self.nr_sigmas,
                        rms=self.rms,
                        asinh_stretch=False if self.stretch_type == "sqrt_stretch" else True
                    )

                    # Skip if no valid proposed boxes
                    if proposed_boxes is None or len(proposed_boxes) == 0:
                        logger.debug(f"Skipping cutout {cutout_index} - no valid proposed boxes")
                        continue
                    
                    # Get next sequential ID (thread-safe)
                    with coco_lock:
                        sample_id = self.next_id
                        self.next_id += 1
                    
                    sample = LoTSS_Negative_GRG_Sample(
                        id=sample_id,
                        image_id=sample_id,
                        category_id=1,
                        ra=cutout.ra,
                        dec=cutout.dec,
                        rgb_image=augmented_data,
                        proposed_boxes=proposed_boxes,
                        proposal_scores=proposal_scores,
                        positions=positions,
                        stretch=self.stretch_type,
                        iscrowd=0,
                        directory=self.save_dir,
                        save_image=True
                    )
                    
                    # Register sample with thread-safe lock
                    with coco_lock:
                        result = sample.register_sample()
                        if result is None:
                            continue
                        coco["images"].append(result['image'])
                    
                    registered_samples += 1

    def _populate_samples(self, coco: dict) -> dict:
        # Pre-create directories to avoid repeated existence checks
        images_dir = os.path.join(self.save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Thread-safe lock for COCO dict updates and ID counter
        coco_lock = threading.Lock()
        
        # Shared counter for sequential IDs (no gaps)
        self.next_id = 1
        
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
                        logger.error(f"Error processing cutout {cutout_index}: {e}", exc_info=True)
                    finally:
                        pbar.update(1)
                        
        return coco

    def _convert_ao_dict_to_segment_dict(
            self,
            ao_dict,
            nr_sigmas: int = 3,
            rms: float = 0.1*1e-3
        ) -> dict[str, Segment]:
        """
        Convert a dictionary of AstroObject instances to a dictionary of Segment instances.
        This is the plug that connects AstroObject with Segment, effectively translating
        the pixel positions of AstroObjects into Segments for segmentation mapping.
        
        :param ao_dict: Dictionary of AstroObject instances
        :return: Dictionary of Segment instances
        """
        segment_dict = {}
        for obj_id, astro_obj in ao_dict.items():
            x_y_pos = astro_obj.get_pixel_positions()
            if isinstance(obj_id, bytes):
                obj_id = obj_id.decode('utf-8')
            segment_dict[obj_id] = Segment(
                x_y_pos,
                nr_sigmas=nr_sigmas,
                rms=rms
            )
        return segment_dict

    def _get_positions(
            self,
            curr_cutout,
            data: np.ndarray,
            nr_sigmas: int = 3,
            rms: float = 0.1*1e-3
        ) -> tuple[np.ndarray, dict[str, int]]:
        # Create CutoutCatalogue for the current cutout
        cutout_cat = CutoutCatalogue(
            catalogue=self.component_catalogue,
            cutout=curr_cutout,
            source_col="Parent_Source"
        )
        # Creates a dict of AstroObject instances for each unique object in the cutout
        ao_dict = cutout_cat.get_astro_objects_from_catalogue()
        segment_dict = self._convert_ao_dict_to_segment_dict(ao_dict, nr_sigmas=nr_sigmas, rms=rms)
        return GRGFinder(seg_dict=segment_dict, data=data).get_positions()
    
    def _get_positions_negative_grg(
            self,
            curr_cutout,
            nr_sigmas: int = 3,
            rms: float = 0.1*1e-3
        ) -> tuple[np.ndarray, dict[str, int]]:
        # Create CutoutCatalogue for the current cutout
        cutout_cat = CutoutCatalogue(
            catalogue=self.component_catalogue,
            cutout=curr_cutout,
            source_col="Parent_Source"
        )
        # Creates a dict of AstroObject instances for each unique object in the cutout
        # unique_objects=False gets every component as a separate AstroObject,
        # which is what we want for search datasets
        ao_dict = cutout_cat.get_astro_objects_from_catalogue(unique_objects=False)
        segment_dict = self._convert_ao_dict_to_segment_dict(ao_dict, nr_sigmas=nr_sigmas, rms=rms)

        positions = {}  # {key: list of (x, y) positions}
        
        for key, segment in segment_dict.items():
            positions[key] = segment.positions[0] # Get just the tuple of (x, y) positions for this component
        return positions
    

class GRGSearchDatasetBuilder(GRGDatasetBuilder):
    def __init__(
            self,
            cutouts: list,
            component_catalogue: pd.DataFrame,
            known_grg_component_names: list[str],
            max_precomputed_islands: int,
            nr_sigmas: int,
            rms: float,
            stretch_type: str,
            segmentation_mode: str,
            workers: int,
            save_dir: str
        ):
        super().__init__(
            cutouts=cutouts,
            component_catalogue=component_catalogue,
            rotation_angles=[0],  # No rotation for search dataset
            crop_size=None,  # No cropping for search dataset
            max_precomputed_islands=max_precomputed_islands,
            nr_sigmas=nr_sigmas,
            rms=rms,
            stretch_type=stretch_type,
            segmentation_mode=segmentation_mode,
            class_ratio=None,  # No negative samples for search dataset
            workers=workers,
            save_dir=save_dir
        )
        self.known_grg_component_names = known_grg_component_names

    def build(self) -> dict:
        """
        Build the COCO dataset by generating samples, categories, and saving to a JSON file.

        :return: Dictionary representing the COCO dataset.
        :rtype: dict
        """
        return CocoDatasetBuilderBase.build(self)

    def _register_sample(self, sample: LoTSS_Search_Sample, coco: dict) -> dict:
        """
        Register a sample into the COCO dataset structure.

        :param sample: The sample to register.
        :type sample: LoTSS_Search_Sample
        :param coco: The COCO dataset dictionary to update.
        :type coco: dict
        :return: Updated COCO dataset dictionary.
        :rtype: dict
        """
        result = sample.register_sample()
        if result is None:
            return coco
        coco["images"].append(result['image'])
        if result['annotation'] is not None:
            coco["annotations"].append(result['annotation'])
        return coco

    def _process_single_cutout_thread(self, cutout, cutout_index, coco, coco_lock):
        """
        Process a single cutout in a thread and register samples immediately.
        For search datasets, we don't have annotations.
        """
        try:
            # Get the data and the positions from the cutout
            data = cutout.get_data()
            positions, grg_positions, grg_positions_dict = self._get_positions(
                cutout,
                nr_sigmas=self.nr_sigmas,
                rms=self.rms
            )
            
            if len(positions) == 0:
                logger.warning(f"Skipping cutout at RA: {cutout.ra}, DEC: {cutout.dec} - no radio components found.")
                return
                
            # Augment and get proposals
            (
                augmented_data,
                proposed_boxes,
                proposal_scores,
            ) = augment_and_get_proposals(
                data=data,
                positions=positions,
                max_precomputed_islands=self.max_precomputed_islands,
                nr_sigmas=self.nr_sigmas,
                rms=self.rms,
                asinh_stretch=False if self.stretch_type == "sqrt_stretch" else True
            )

            # Annotate if any known GRG components are present (for evaluation purposes, not used in training)
            if grg_positions:
                grg_segm, bboxes = annotate(
                    grg_positions=grg_positions,
                    data=data,
                    nr_sigmas=self.nr_sigmas,
                    rms=self.rms,
                )
            else:
                grg_segm, bboxes = None, None
            

            # Skip if no valid proposed boxes
            if proposed_boxes is None or len(proposed_boxes) == 0:
                logger.debug(f"Skipping cutout {cutout_index} - no valid proposed boxes")
                return
            
            # Get next sequential ID (thread-safe)
            with coco_lock:
                sample_id = self.next_id
                self.next_id += 1
            
            sample = LoTSS_Search_Sample(
                id=sample_id,
                image_id=sample_id,
                grg_segmentation=grg_segm,
                grg_bboxes=bboxes,
                category_id=1,
                ra=cutout.ra,
                dec=cutout.dec,
                rgb_image=augmented_data,
                proposed_boxes=proposed_boxes,
                proposal_scores=proposal_scores,
                positions=positions,
                grg_positions=grg_positions_dict,
                seg_mode=self.segmentation_mode,
                stretch=self.stretch_type,
                iscrowd=0,
                directory=self.save_dir,
                save_image=True
            )
            
            # Register sample with thread-safe lock
            with coco_lock:
                coco = self._register_sample(sample, coco)

            # Free memory
            del data
            del positions
            del augmented_data
            del proposed_boxes
            del proposal_scores
            gc.collect()
            
        except Exception as e:
            print(f"\n!!! ERROR in cutout {cutout_index} !!!")
            print(f"Cutout RA: {cutout.ra}, DEC: {cutout.dec}")
            if hasattr(cutout, 'mosaic'):
                print(f"Mosaic: {cutout.mosaic.field_name if hasattr(cutout.mosaic, 'field_name') else 'unknown'}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!\n")
            logger.error(f"Error in thread processing cutout {cutout_index}: {e}", exc_info=True)

    def _get_positions(
            self,
            curr_cutout,
            nr_sigmas: int = 3,
            rms: float = 0.1*1e-3
        ) -> tuple[np.ndarray, dict[str, int]]:
        # Create CutoutCatalogue for the current cutout
        cutout_cat = CutoutCatalogue(
            catalogue=self.component_catalogue,
            cutout=curr_cutout,
            source_col="Parent_Source"
        )
        # Creates a dict of AstroObject instances for each unique object in the cutout
        # unique_objects=False gets every component as a separate AstroObject,
        # which is what we want for search datasets
        ao_dict = cutout_cat.get_astro_objects_from_catalogue(unique_objects=False)
        segment_dict = self._convert_ao_dict_to_segment_dict(ao_dict, nr_sigmas=nr_sigmas, rms=rms)

        positions = {}  # {key: list of (x, y) positions}
        
        grg_positions = []
        grg_positions_dict = {}  # {key: list of (x, y) positions} for known GRG components only
        for key, segment in segment_dict.items():
            positions[key] = segment.positions[0] # Get just the tuple of (x, y) positions for this component
            if self._known_grg_component(key):
                grg_positions.append(segment.positions[0])
                grg_positions_dict[key] = segment.positions[0]
        return positions, grg_positions, grg_positions_dict

    def _known_grg_component(self, component_name: str) -> bool:
        return component_name in self.known_grg_component_names
    