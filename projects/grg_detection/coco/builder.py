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

from .sample import LoTSS_Sample
from .category import LoTSS_GRG_CocoCategory

# Append project path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from grg_detection.annotations import annotate_and_augment, GRGFinder

sys.path.append('/home/penchev/strw_lofar_data_utils/')
from src.core.cutout_maker.cutout_catalogue import CutoutCatalogue


logger = logging.getLogger("GRGDatasetBuilder")


GRG_CATEGORY = LoTSS_GRG_CocoCategory(id=1, name="GRG")


class GRGDatasetBuilder(CocoDatasetBuilderBase):
    def __init__(
            self,
            cutouts: list,
            component_catalogue: pd.DataFrame,
            dataset_type: str,
            rotation_angles: list[int],
            crop_size: int,
            max_precomputed_islands: int,
            rms: float,
            stretch_type: str,
            save_dir: str
        ):
        self.cutouts = cutouts
        self.component_catalogue = component_catalogue

        # Make sure save directory exists
        self.save_dir = os.path.join(save_dir, dataset_type)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.dataset_type = dataset_type
        self.rotation_angles = rotation_angles
        self.crop_size = crop_size
        self.max_precomputed_islands = max_precomputed_islands
        self.rms = rms
        self.stretch_type = stretch_type

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
            # Get the data and the positions from the cutout
            data = cutout.get_data()
            grg_positions, all_component_positions = self._get_positions(cutout, data)
            
            if grg_positions is None:
                logger.warning(f"Skipping cutout at RA: {cutout.ra}, DEC: {cutout.dec} - no GRG annotation found.")
                return
                
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
                rms=self.rms,
                asinh_stretch=False if self.stretch_type == "sqrt_stretch" else True
            )

            # Create and register samples for all rotations
            for angle_index, angle in enumerate(self.rotation_angles):
                # Deterministic ID calculation
                sample_id = cutout_index * len(self.rotation_angles) + angle_index + 1
                
                sample = LoTSS_Sample(
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
                    rotation_angle=angle,
                    stretch=self.stretch_type,
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
            
        except Exception as e:
            logger.error(f"Error in thread processing cutout {cutout_index}: {e}", exc_info=True)

    def _populate_samples(self, coco: dict) -> dict:
        # Pre-create directories to avoid repeated existence checks
        images_dir = os.path.join(self.save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Thread-safe lock for COCO dict updates
        coco_lock = threading.Lock()
        
        # Process cutouts in parallel with ThreadPoolExecutor
        max_workers = min(os.cpu_count() or 1, 10)  # Limit to 10 workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            with tqdm(total=len(self.cutouts), desc=f"Generating LoTSS Samples for COCO '{self.dataset_type}' Dataset") as pbar:
                for future in as_completed(future_to_index):
                    cutout_index = future_to_index[future]
                    try:
                        future.result()  # Samples already registered inside thread
                    except Exception as e:
                        logger.error(f"Error processing cutout {cutout_index}: {e}", exc_info=True)
                    finally:
                        pbar.update(1)
                        
        return coco

    def _convert_ao_dict_to_segment_dict(self, ao_dict):
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
            segment_dict[obj_id] = Segment(x_y_pos, nr_sigmas=5)
        return segment_dict

    def _get_positions(self, curr_cutout, data: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        # Create CutoutCatalogue for the current cutout
        cutout_cat = CutoutCatalogue(
            catalogue=self.component_catalogue,
            cutout=curr_cutout,
            source_col="Parent_Source"
        )
        # Creates a dict of AstroObject instances for each unique object in the cutout
        ao_dict = cutout_cat.get_astro_objects_from_catalogue()
        segment_dict = self._convert_ao_dict_to_segment_dict(ao_dict)
        return GRGFinder(seg_dict=segment_dict, data=data).get_positions()
    