from astropy.table import Table
import sys
import timeit
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import argparse

from astro_datatools import setup_logging
from astro_datatools.lotss_annotations import Segment

from grg_detection.coco import GRGSearchDatasetBuilder
from grg_detection.annotations import GRGFinder

from strw_lofar_data_utils.core.cutout_maker import SourceBlob, CutoutCatalogue
from strw_lofar_data_utils.core.mosaic_crawler import DR2Crawler
from strw_lofar_data_utils.pipelines import generate_cutouts


# Setup logger
logger = logging.getLogger("search_dataset_pipeline")


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    :param config_path: Path to the YAML configuration file.
    :type config_path: str
    :return: Dictionary containing configuration parameters.
    :rtype: dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_annotation_component_names(
    giants_catalog: pd.DataFrame,
    cutout_size: int,
    mosaics_to_crawl: list[str],
    component_catalogue: pd.DataFrame,
    nr_sigmas: int = 3,
    rms: float = 0.1*1e-3
    ):    
    # Get RA and DEC for the giants and create cutouts
    ra_dec_list = list(
        zip(giants_catalog["RAJ2000"].values, giants_catalog["DEJ2000"].values)
    )
    logger.info("Generating cutouts of known GRGs...")
    giants_cutouts = generate_cutouts(
        ra_dec_list=ra_dec_list,
        size_pixels=cutout_size,
        save=False
    )

    # Get the cutouts in the specified mosaics to crawl
    cutouts = [cut for cut in giants_cutouts if cut.mosaic.field_name in mosaics_to_crawl]
    logger.info(f"Generated {len(cutouts)} cutouts containing known GRGs from the giants catalog that are in the specified mosaics to crawl.")

    def get_component_names(
            curr_cutout,
            component_catalogue,
            data: np.ndarray,
            nr_sigmas: int = 3,
            rms: float = 0.1*1e-3
        ) -> tuple[np.ndarray, dict[str, int]]:
        def _convert_ao_dict_to_segment_dict(
                sb_dict: dict[str, SourceBlob],
                nr_sigmas: int = 3,
                rms: float = 0.1*1e-3
            ) -> dict[str, Segment]:
            """
            Convert a dictionary of SourceBlob instances to a dictionary of Segment instances.
            This is the plug that connects SourceBlob with Segment, effectively translating
            the pixel positions of SourceBlobs into Segments for segmentation mapping.
            
            :param sb_dict: Dictionary of SourceBlob instances
            :return: Dictionary of Segment instances
            """
            segment_dict = {}
            for obj_id, source_blob in sb_dict.items():
                x_y_pos = source_blob.get_pixel_positions()
                if isinstance(obj_id, bytes):
                    obj_id = obj_id.decode('utf-8')
                segment_dict[obj_id] = Segment(
                    x_y_pos,
                    nr_sigmas=nr_sigmas,
                    rms=rms
                )
            return segment_dict
    
        def _match_components_to_grg_positions(grg_positions, position: list[tuple[int, int]]):
            for grg_pos in grg_positions:
                if np.array_equal(grg_pos, position[0]):
                    return True
            return False
        
        # Create CutoutCatalogue for the current cutout
        cutout_cat = CutoutCatalogue(
            catalogue=component_catalogue,
            cutout=curr_cutout,
            source_col="Parent_Source"
        )

        # Creates a dict of SourceBlob instances for each unique object in the cutout
        sb_dict = cutout_cat.get_source_blobs_from_catalogue()
        segment_dict = _convert_ao_dict_to_segment_dict(sb_dict, nr_sigmas=nr_sigmas, rms=rms)

        # Use GRGFinder to get the positions of the GRG components in the cutout based on the segments
        grg_positions, _ = GRGFinder(seg_dict=segment_dict, data=data).get_positions()

        # If no GRG positions are found, return an empty list
        # to avoid errors in the annotation procedure and to allow the pipeline to
        # continue generating cutouts from the specified mosaics without annotation
        if grg_positions is False:
            return []

        component_names = []
        sb_dict2 = cutout_cat.get_source_blobs_from_catalogue(unique_objects=False)
        for obj, source_blob in sb_dict2.items():
            position = source_blob.get_pixel_positions() # list[tuple[int, int]]
            if _match_components_to_grg_positions(grg_positions, position):
                if isinstance(obj, bytes):
                    obj = obj.decode('utf-8')
                component_names.append(obj)
            else:
                continue

        if len(component_names) == len(grg_positions):
            return component_names
        else:
            logger.warning(
                f"Warning: Number of matched component names ({len(component_names)}) does not match number of GRG positions ({len(grg_positions)}). "
                f"This happens for cutout with center RA: {curr_cutout.ra}, DEC: {curr_cutout.dec} in mosaic {curr_cutout.mosaic.field_name}. "
                "Returning matched component names anyway.")
            return component_names

    component_names = []
    for cut in cutouts:
        data = cut.get_data()
        curr_components = get_component_names(
            curr_cutout=cut,
            component_catalogue=component_catalogue,
            data=data,
            nr_sigmas=nr_sigmas,
            rms=rms
        )
        component_names.extend(curr_components)

    logger.info(f"Total matched component names: {len(component_names)}")
    return component_names

def main(config_path: str):
    start_time = timeit.default_timer()
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration values for PATHS
    COMPONENT_CATALOGUE_FILEPATH = config['PATHS']['COMPONENT_CATALOGUE_FILEPATH']
    DATASET_SAVE_DIR = config['PATHS']['SAVE_DIR']
    if 'GIANTS_CATALOG_FILEPATH' in config['PATHS']: # Extract GIANTS_CATALOG_FILEPATH if it exists in the config, otherwise set to None
        GIANTS_CATALOG_FILEPATH = config['PATHS']['GIANTS_CATALOG_FILEPATH']
    else:
        GIANTS_CATALOG_FILEPATH = None
    
    # Extract configuration values for CUTOUT_PARAMS
    CUTOUT_SIZE = config['CUTOUT_PARAMS']['CUTOUT_SIZE']
    RMS = config['CUTOUT_PARAMS']['RMS']
    NR_SIGMAS = config['CUTOUT_PARAMS']['NR_SIGMAS']
    
    # Extract configuration values for AUGMENTATION
    STRETCH_TYPE = config['AUGMENTATION']['STRETCH_TYPE']
    MAX_PRECOMPUTED_ISLANDS = config['AUGMENTATION']['MAX_PRECOMPUTED_ISLANDS']
    SEGMENTATION_MODE = config['AUGMENTATION']['SEGMENTATION_MODE']
    
    # Extract configuration values for MOSAIC_TO_CRAWL
    MOSAICS_TO_CRAWL = config['MOSAICS_TO_CRAWL']['MOSAICS_NAME']
    STRIDE = config['MOSAICS_TO_CRAWL']['STRIDE']
    ANNOTATE_POSITIVES = config['MOSAICS_TO_CRAWL']['ANNOTATE_POSITIVES']

    # Extract configuration values for PIPELINE
    WORKERS = config['PIPELINE']['WORKERS']

    # Setup logging
    log_filepath = os.path.join(DATASET_SAVE_DIR, "search_dataset_pipeline.log")
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

    logger.info("Starting search dataset generation pipeline.")
    logger.info(f"Using configuration from: {config_path}")

    # Load the component catalogue
    logger.info(f"Loading component catalog from: {COMPONENT_CATALOGUE_FILEPATH}")
    COMPONENT_CATALOGUE_TABLE = Table.read(COMPONENT_CATALOGUE_FILEPATH)
    COMPONENT_CATALOGUE = COMPONENT_CATALOGUE_TABLE.to_pandas()

    # Generate the component names for annotation based on the known GRGs in the giants catalog
    # if ANNOTATE_POSITIVES is True,
    # otherwise skip this step and set COMPONENT_DICT to an empty dictionary
    if ANNOTATE_POSITIVES:
        if GIANTS_CATALOG_FILEPATH is None:
            raise ValueError(
                "ANNOTATE_POSITIVES is True but GIANTS_CATALOG_FILEPATH is not provided in the config. "
                "If you want to annotate cutouts with known GRGs, please provide the GIANTS_CATALOG_FILEPATH in the config. "
                "If you do not want to annotate cutouts with known GRGs, set ANNOTATE_POSITIVES to False in the config."
            )
        giants_catalog = pd.read_csv(GIANTS_CATALOG_FILEPATH)

        # Remove the giants from "Hardcastle et al. 2023" since they are used for training
        # TODO: In the future this might need to be changed!
        giants_catalog = giants_catalog[giants_catalog['FirstDisc'] != 'Hardcastle et al. 2023'].reset_index(drop=True)
        logger.warning(
            "[BE CAREFUL AND DO NOT IGNORE THIS MESSAGE!]: "
            "Removed giants from 'Hardcastle et al. 2023' in the giants catalog since they are used for training. "
        )
        COMPONENT_NAMES = get_annotation_component_names(
            giants_catalog=giants_catalog,
            cutout_size=CUTOUT_SIZE,
            mosaics_to_crawl=MOSAICS_TO_CRAWL,
            component_catalogue=COMPONENT_CATALOGUE,
            nr_sigmas=NR_SIGMAS,
            rms=RMS
        )
    else:
        logger.info("ANNOTATE_POSITIVES is False, skipping the annotation of known GRGs and generating cutouts from the specified mosaics to crawl without annotation.")
        COMPONENT_NAMES = [] # Empty list since we are not annotating known GRGs

    logger.info(f"Crawling the following mosaics: {MOSAICS_TO_CRAWL} with stride: {STRIDE} to generate cutouts for the search dataset.")
    
    # Crawl the specified mosaics and generate cutouts
    cutouts = []
    for mosaic in tqdm(MOSAICS_TO_CRAWL, desc="Crawling mosaics"):
        crawler = DR2Crawler(mosaic, CUTOUT_SIZE, STRIDE, verbose=False)
        results = crawler.crawl()
        cutouts.extend(results)

    logger.info(f"Finished crawling mosaics. Total cutouts generated: {len(cutouts)}")
    logger.info("Building dataset with all cutouts...")
    GRGSearchDatasetBuilder(
        cutouts=cutouts,
        component_catalogue=COMPONENT_CATALOGUE,
        known_grg_component_names=COMPONENT_NAMES,
        max_precomputed_islands=MAX_PRECOMPUTED_ISLANDS,
        nr_sigmas=NR_SIGMAS,
        rms=RMS,
        stretch_type=STRETCH_TYPE,
        segmentation_mode=SEGMENTATION_MODE,
        workers=WORKERS,
        save_dir=DATASET_SAVE_DIR
    ).build()
    logger.info(f"Finished building the search dataset.")

    elapsed_time = timeit.default_timer() - start_time
    logger.info(f"Search dataset generation pipeline completed successfully in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate COCO dataset to search for GRGs in LoTSS mosaics."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../configs/search_dataset.yaml"),
        help="Path to YAML configuration file (default: ../configs/search_dataset.yaml)"
    )
    args = parser.parse_args()
    
    main(args.config)
