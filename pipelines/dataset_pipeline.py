from astropy.table import Table
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import gc
from copy import deepcopy
import logging

sys.path.append('/home/penchev/astro-datatools/')
from astro_datatools import setup_logging
from astro_datatools.augment import RotateAugment, LotssToRGBAugment
from astro_datatools.lotss_annotations import Segment
from astro_datatools.lotss_annotations.grg_full_annotation import GRGFullAnnotation
from astro_datatools.lotss_annotations.precompute_proposals import PrecomputeProposals
from astro_datatools.core.datasets.coco.builder import CocoDatasetBuilderBase
from astro_datatools.core.datasets.coco.lotss_sample import LoTSS_Sample
from astro_datatools.core.datasets.coco.category import LoTSS_GRG_CocoCategory

sys.path.append('/home/penchev/strw_lofar_data_utils/')
from src.pipelines import generate_cutouts
from src.core.cutout_maker.cutout_catalogue import CutoutCatalogue

logger = logging.getLogger(__name__)

# Load the discovered giants catalogue
GIANTS_CATALOG_FILEPATH = "/home/penchev/strw_lofar_data_utils/data/discovered_giants_catalogue.csv"
GIANTS_CATALOG = pd.read_csv(GIANTS_CATALOG_FILEPATH)

# Load the component catalogue
COMPONENT_CATALOGUE_FILEPATH = "/net/vdesk/data2/penchev/project_data/data/combined_cat/combined-components-v0.8.fits"
COMPONENT_CATALOGUE_TABLE = Table.read(COMPONENT_CATALOGUE_FILEPATH) # Read the catalogue as an Astropy Table
COMPONENT_CATALOGUE = COMPONENT_CATALOGUE_TABLE.to_pandas() # Convert to Pandas DataFrame for easier handling

# Directory to save the dataset
DATASET_SAVE_DIR = "/net/vdesk/data2/penchev/project_data/full-dataset/"

# Get RA and DEC for the first 100 giants
RA_DEC_LIST = list(
    zip(GIANTS_CATALOG["RAJ2000"].values, GIANTS_CATALOG["DEJ2000"].values)
)
# RA_DEC_LIST = RA_DEC_LIST[:100] # Chose only first 100 for testing

CUTOUT_SIZE = 425 # pixels - this gives us 300 pixels after rotation of 45 degrees
CROP_SIZE = 300 # pixels - final size after rotation and cropping

# Rotation angles
ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# Stretch type
STRETCH_TYPE = "sqrt_stretch"

# RMS noise level (in Jy/beam)
RMS = 0.1 * 1e-3 # 0.1 mJy/beam

# Dataset splits
DATASET_SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

MAX_PRECOMPUTED_ISLANDS = 10 # To limit combinatorial explosion in proposals

GRG_CATEGORY = LoTSS_GRG_CocoCategory(id=1, name="GRG")


class LotssDatasetBuilder(CocoDatasetBuilderBase):
    def __init__(self, cutouts: list, component_catalogue: pd.DataFrame, dataset_type: str, save_dir: str):
        self.cutouts = cutouts
        self.component_catalogue = component_catalogue

        # Make sure save directory exists
        self.save_dir = os.path.join(save_dir, dataset_type)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.rotator = RotateAugment(
            angles=ROTATION_ANGLES,
            dynamic_cropping=False,
            specific_crop_size=(CROP_SIZE, CROP_SIZE)
        )
        self.rgb_augmenter = LotssToRGBAugment(
            rms_noise=RMS,
            asinh_stretch=False if STRETCH_TYPE == "sqrt_stretch" else True
        )

        self.dataset_type = dataset_type

    def _get_filepath(self) -> str:
        return os.path.join(self.save_dir, "annotations.json")
        
    def _add_categories(self, coco: dict) -> dict:
        coco['categories'].append(GRG_CATEGORY.to_dict())
        return coco

    def _populate_samples(self, coco: dict) -> dict:
        id_ = 1
        
        # Pre-create directories to avoid repeated existence checks
        images_dir = os.path.join(self.save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for cutout in tqdm(self.cutouts, desc=f"Generating LoTSS Samples for COCO {self.dataset_type} Dataset"):
            # Get annotations
            data = cutout.get_data()
            grg_seg, grg_bbox, seg_map = self._annotate(cutout, data)
            if grg_seg is None or grg_bbox is None:
                logger.warning(f"Skipping cutout at RA: {cutout.ra}, DEC: {cutout.dec} - no GRG annotation found.")
                continue
            proposed_boxes, proposal_scores = PrecomputeProposals(
                seg_map, max_islands=MAX_PRECOMPUTED_ISLANDS
            ).precompute(return_scores=True)

            # Apply rotations (already optimized)
            rotated_data = self.rotator.augment(data) # (B, H, W)
            rotated_grg_segs = self.rotator.augment(grg_seg) # (B, H, W)
            rotated_grg_bboxes = self.rotator.augment_bbox(
                grg_bbox, original_h=CUTOUT_SIZE, original_w=CUTOUT_SIZE
            )
            rotated_proposed_boxes = self.rotator.augment_proposal_boxes(
                proposed_boxes, original_h=CUTOUT_SIZE, original_w=CUTOUT_SIZE
            )

            # Get RGB images (vectorized for all rotations at once)
            rgb_rotated_data = self.rgb_augmenter.augment(rotated_data) # (B, C, H, W)

            # Create samples for all rotations
            for angle_index, angle in enumerate(ROTATION_ANGLES):
                sample = LoTSS_Sample(
                    id=id_ ,
                    image_id=id_,
                    category_id=1, # Only one category for GRGs
                    ra=cutout.ra,
                    dec=cutout.dec,
                    rgb_image=rgb_rotated_data[angle_index], # (C, H, W)
                    full_annotation=(rotated_grg_segs[angle_index], rotated_grg_bboxes[angle_index]),
                    proposed_boxes=rotated_proposed_boxes[angle_index],
                    proposal_scores=proposal_scores,
                    rotated=True if angle != 0 else False,
                    rotation_angle=angle,
                    stretch=STRETCH_TYPE,
                    reprojected=False,
                    old_redshift=None,
                    new_redshift=None,
                    iscrowd=0,
                    directory=self.save_dir,
                    save_image=True
                )
                coco = self._register_sample(sample, coco)
                id_ += 1

            # Free memory
            del data
            del grg_seg, grg_bbox, seg_map
            del proposed_boxes, proposal_scores
            del rotated_data
            del rotated_grg_segs
            del rotated_grg_bboxes
            del rotated_proposed_boxes
            del rgb_rotated_data
            gc.collect()
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

    def _annotate(self, curr_cutout, data: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        # Create CutoutCatalogue for the current cutout
        # This is needed to get the objects within the cutout
        # and is implemented in the strw_lofar_data_utils package.
        # It is the main way to link cutouts with catalog objects.
        cutout_cat = CutoutCatalogue(
            catalogue=COMPONENT_CATALOGUE,
            cutout=curr_cutout,
            source_col="Parent_Source"
        )

        # Creates a dict of AstroObject instances for each unique object in the cutout
        ao_dict = cutout_cat.get_astro_objects_from_catalogue()
        segment_dict = self._convert_ao_dict_to_segment_dict(ao_dict)
        return GRGFullAnnotation(seg_dict=segment_dict, data=data).get_annotation()

def main():
    # Setup logging
    log_filepath = os.path.join(DATASET_SAVE_DIR, "dataset_pipeline.log")
    setup_logging(log_file=log_filepath)
    
    # Redirect stdout and stderr to also write to the log file
    # This captures tqdm progress bars
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file_handle = open(log_filepath, 'a')
    sys.stdout = TeeOutput(sys.stdout, log_file_handle)
    sys.stderr = TeeOutput(sys.stderr, log_file_handle)

    logger.info("Starting dataset generation pipeline.")

    # Create the cutouts
    logger.info("Generating cutouts...")
    cutouts = generate_cutouts(
        ra_dec_list=RA_DEC_LIST,
        size_pixels = CUTOUT_SIZE,
        save=False
    )
    # cutouts = cutouts[:10] # For testing, use only first 10 cutouts
    logger.info(f"Generated {len(cutouts)} cutouts.")

    logger.info("Splitting cutouts into train, val, test sets...")
    # Split the cutouts into train, val, test randomly
    np.random.shuffle(cutouts)
    num_cutouts = len(cutouts)
    train_end = int(DATASET_SPLITS["train"] * num_cutouts)
    val_end = train_end + int(DATASET_SPLITS["val"] * num_cutouts)
    cutouts_splits = {
        "train": cutouts[:train_end],
        "val": cutouts[train_end:val_end],
        "test": cutouts[val_end:]
    }

    for key, cutouts_split in cutouts_splits.items():
        LotssDatasetBuilder(
            cutouts=cutouts_split,
            component_catalogue=COMPONENT_CATALOGUE,
            dataset_type=key,
            save_dir=DATASET_SAVE_DIR
        ).build()
        logger.info(f"Finished building {key} dataset.")


if __name__ == "__main__":
    main()
