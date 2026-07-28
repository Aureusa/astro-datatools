from astropy.table import Table
import timeit
import pandas as pd
import numpy as np
import logging
import yaml
import argparse

from astro_datatools.logger import setup_logging

from b2s_masked_rcnn.coco import B2SDatasetBuilder

from strw_lofar_data_utils.pipelines import generate_cutouts

import os


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


# Default dataset splits (not typically changed)
DATASET_SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}


def main(config_path: str):
    start_time = timeit.default_timer()
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration values for PATHS
    GIANTS_CATALOG_FILEPATH = config['PATHS']['GIANTS_CATALOG_FILEPATH']
    COMPONENT_CATALOGUE_FILEPATH = config['PATHS']['COMPONENT_CATALOGUE_FILEPATH']
    DATASET_SAVE_DIR = config['PATHS']['SAVE_DIR']
    
    # Extract configuration values for CUTOUT_PARAMS
    CUTOUT_SIZE = config['CUTOUT_PARAMS']['CUTOUT_SIZE']
    CROP_SIZE = config['CUTOUT_PARAMS']['CROP_SIZE']
    RMS = config['CUTOUT_PARAMS']['RMS']
    NR_SIGMAS = config['CUTOUT_PARAMS']['NR_SIGMAS']

    # Extract configuration values for AUGMENTATION
    CLASS_RATIO = config['AUGMENTATION']['CLASS_RATIO']
    ROTATION_ANGLES = config['AUGMENTATION']['ROTATION_ANGLES']
    STRETCH_TYPE = config['AUGMENTATION']['STRETCH_TYPE']
    MAX_PRECOMPUTED_ISLANDS = config['AUGMENTATION']['MAX_PRECOMPUTED_ISLANDS']

    # Extract configuration values for PIPELINE
    WORKERS = config['PIPELINE']['WORKERS']
    LIMIT_RADEC_LIST = config['PIPELINE']['LIMIT_RADEC_LIST']
    LIMIT_CUTOUTS = config['PIPELINE']['LIMIT_CUTOUTS']

    MULTICLASS = config['ANNOTATIONS']['MULTICLASS']
    if MULTICLASS:
        INVALID_LABEL = config['ANNOTATIONS']['INVALID_LABEL']
        SCS_LABEL = config['ANNOTATIONS']['SCS_LABEL']
        MCS_LABEL = config['ANNOTATIONS']['MCS_LABEL']
    
    # Setup logging
    log_filepath = os.path.join(DATASET_SAVE_DIR, "dataset_pipeline.log")
    setup_logging(log_file=log_filepath)
    logger = setup_logging(name="b2s_masked_rcnn.pipelines.dataset_pipeline", log_file=log_filepath)

    # Check for RGZ specific
    RGZ_RADEC_LIST_FILEPATH = config['PATHS'].get('RGZ_RADEC_LIST_FILEPATH', None)
    if RGZ_RADEC_LIST_FILEPATH is not None:
        logger.info(f"RGZ-specific RA/DEC list file provided: {RGZ_RADEC_LIST_FILEPATH}")
        logger.info("Loading RGZ-specific RA/DEC list...")
        RGZ_RA_DEC_LIST = pd.read_csv(RGZ_RADEC_LIST_FILEPATH)
        RGZ_RA_DEC_LIST = list(zip(RGZ_RA_DEC_LIST["RA"].values, RGZ_RA_DEC_LIST["DEC"].values))
        logger.info(f"Loaded {len(RGZ_RA_DEC_LIST)} RA/DEC pairs from RGZ-specific list.")
    else:
        RGZ_RA_DEC_LIST = None
    
    logger.info("Starting dataset generation pipeline.")
    logger.info(f"Using configuration from: {config_path}")
    
    # Load the component catalogue
    logger.info(f"Loading component catalog from: {COMPONENT_CATALOGUE_FILEPATH}")
    COMPONENT_CATALOGUE_TABLE = Table.read(COMPONENT_CATALOGUE_FILEPATH)
    COMPONENT_CATALOGUE = COMPONENT_CATALOGUE_TABLE.to_pandas()
    
    # Get RA and DEC
    if RGZ_RA_DEC_LIST is not None: # RGZ
        RA_DEC_LIST = RGZ_RA_DEC_LIST
        logger.info(f"Using RGZ-specific RA/DEC list with {len(RA_DEC_LIST)} cutouts.")
    else: # Giants catalog
        # Load the discovered giants catalogue
        logger.info(f"Loading giants catalog from: {GIANTS_CATALOG_FILEPATH}")
        GIANTS_CATALOG = pd.read_csv(GIANTS_CATALOG_FILEPATH)

        okay_ish = [
            "Hardcastle et al. 2023",
            "Dabhade et al. 2020 / prior",
            "Mahato et al. 2021",
            "Tang et al. 2020",
            "Bassani et al. 2020",
            "Masini et al. 2021",
        ]
        GIANTS_CATALOG = GIANTS_CATALOG[GIANTS_CATALOG["FirstDisc"].isin(okay_ish)]

        # From the GIANTS CATALOG get only 'Hardcastle et al. 2023' from the 'FirstDisc' column
        # GIANTS_CATALOG = GIANTS_CATALOG[GIANTS_CATALOG['FirstDisc'] == 'Hardcastle et al. 2023'].reset_index(drop=True)
        cats_used_str = ""
        for cat in okay_ish:
            cats_used_str += f"\n- `{cat}`"
        logger.warning(
            "[BE CAREFUL AND DO NOT IGNORE THIS MESSAGE!]: " + "\n"
            "Using only the giants from:" + 
            f"{cats_used_str}"       
        ) # TODO: REMOVE THIS WARNING IN THE FUTURE WHEN THE GIANTS CATALOG IS FINALIZED AND WE WANT TO USE ALL THE GIANTS!
        
        RA_DEC_LIST = list(
            zip(GIANTS_CATALOG["RAJ2000"].values, GIANTS_CATALOG["DEJ2000"].values)
        )

    # Limit the RA/DEC list if specified in the config
    if LIMIT_RADEC_LIST is not None:
        logger.info(f"Limiting RA/DEC list to the first {LIMIT_RADEC_LIST} entries.")
        RA_DEC_LIST = RA_DEC_LIST[:LIMIT_RADEC_LIST]
    else:
        logger.info(f"Processing the full RA/DEC list with {len(RA_DEC_LIST)} cutouts.")

    # Create the cutouts
    logger.info("Generating cutouts...")
    cutouts = generate_cutouts(
        ra_dec_list=RA_DEC_LIST,
        size_pixels = CUTOUT_SIZE,
        save=False
    )
    cutouts = cutouts[:LIMIT_CUTOUTS] if LIMIT_CUTOUTS is not None else cutouts
    cutouts.sort(key=lambda cutout: (cutout.mosaic.field_name, cutout.ra, cutout.dec))
    logger.info(f"Generated {len(cutouts)} cutouts.")

    logger.info("Building dataset with all cutouts...")
    B2SDatasetBuilder(
        cutouts=cutouts,
        component_catalogue=COMPONENT_CATALOGUE,
        rotation_angles=ROTATION_ANGLES,
        crop_size=CROP_SIZE,
        max_precomputed_islands=MAX_PRECOMPUTED_ISLANDS,
        nr_sigmas=NR_SIGMAS,
        rms=RMS,
        stretch_type=STRETCH_TYPE,
        multiclass=MULTICLASS,
        labels={
            "invalid": INVALID_LABEL,
            "scs": SCS_LABEL,
            "mcs": MCS_LABEL
        } if MULTICLASS else None,
        class_ratio=CLASS_RATIO,
        workers=WORKERS,
        save_dir=DATASET_SAVE_DIR
    ).build()
    logger.info(f"Finished building the dataset.")

    elapsed_time = timeit.default_timer() - start_time
    if elapsed_time < 60: # less than a minute
        logger.info(f"Dataset generation pipeline completed successfully in {elapsed_time:.2f} seconds.")
    elif elapsed_time < 3600: # less than an hour
        minutes, seconds = divmod(elapsed_time, 60)
        logger.info(f"Dataset generation pipeline completed successfully in {int(minutes)} minutes and {seconds:.2f} seconds.")
    else:
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Dataset generation pipeline completed successfully in {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate COCO dataset for GRG detection from LoTSS data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../configs/dataset_pipeline.yaml"),
        help="Path to YAML configuration file (default: ../configs/dataset_pipeline.yaml)"
    )
    args = parser.parse_args()
    
    main(args.config)
