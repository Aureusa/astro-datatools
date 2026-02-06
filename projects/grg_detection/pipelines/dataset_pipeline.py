from astropy.table import Table
import sys
import timeit
import pandas as pd
import numpy as np
import os
import logging
import yaml
import argparse

# Append astro-datatools path for imports
sys.path.append('/home/penchev/astro-datatools/')
from astro_datatools import setup_logging

# Append project path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from grg_detection.coco import GRGDatasetBuilder

# Append strw_lofar_data_utils path for imports
sys.path.append('/home/penchev/strw_lofar_data_utils/')
from src.pipelines import generate_cutouts

# Setup logger
logger = logging.getLogger("dataset_generation_pipeline")


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
    
    # Extract configuration values
    GIANTS_CATALOG_FILEPATH = config['PATHS']['GIANTS_CATALOG_FILEPATH']
    COMPONENT_CATALOGUE_FILEPATH = config['PATHS']['COMPONENT_CATALOGUE_FILEPATH']
    DATASET_SAVE_DIR = config['PATHS']['SAVE_DIR']
    
    CUTOUT_SIZE = config['CUTOUT_PARAMS']['CUTOUT_SIZE']
    CROP_SIZE = config['CUTOUT_PARAMS']['CROP_SIZE']
    RMS = config['CUTOUT_PARAMS']['RMS']
    NR_SIGMAS = config['CUTOUT_PARAMS']['NR_SIGMAS']
    ROTATION_ANGLES = config['AUGMENTATION']['ROTATION_ANGLES']
    STRETCH_TYPE = config['AUGMENTATION']['STRETCH_TYPE']
    MAX_PRECOMPUTED_ISLANDS = config['AUGMENTATION']['MAX_PRECOMPUTED_ISLANDS']

    SEGMENTATION_MODE = config['ANNOTATIONS']['SEGMENTATION_MODE']
    
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
    logger.info(f"Using configuration from: {config_path}")
    
    # Load the discovered giants catalogue
    logger.info(f"Loading giants catalog from: {GIANTS_CATALOG_FILEPATH}")
    GIANTS_CATALOG = pd.read_csv(GIANTS_CATALOG_FILEPATH)

    # From the GIANTS CATALOG get only 'RGZ (Hardcastle et al. 2023)' from the 'Ref' column
    GIANTS_CATALOG = GIANTS_CATALOG[GIANTS_CATALOG['Ref'] == 'RGZ (Hardcastle et al. 2023)'].reset_index(drop=True)
    
    # Load the component catalogue
    logger.info(f"Loading component catalog from: {COMPONENT_CATALOGUE_FILEPATH}")
    COMPONENT_CATALOGUE_TABLE = Table.read(COMPONENT_CATALOGUE_FILEPATH)
    COMPONENT_CATALOGUE = COMPONENT_CATALOGUE_TABLE.to_pandas()
    
    # Get RA and DEC for the giants
    RA_DEC_LIST = list(
        zip(GIANTS_CATALOG["RAJ2000"].values, GIANTS_CATALOG["DEJ2000"].values)
    )
    RA_DEC_LIST = RA_DEC_LIST[:1500]  # TODO: Make this configurable or remove for production
    logger.info(f"Processing {len(RA_DEC_LIST)} sources")

    # Create the cutouts
    logger.info("Generating cutouts...")
    cutouts = generate_cutouts(
        ra_dec_list=RA_DEC_LIST,
        size_pixels = CUTOUT_SIZE,
        save=False
    )
    cutouts = cutouts[:1000] # For testing, use only first 100 cutouts
    logger.info(f"Generated {len(cutouts)} cutouts.")

    logger.info("Building dataset with all cutouts...")
    GRGDatasetBuilder(
        cutouts=cutouts,
        component_catalogue=COMPONENT_CATALOGUE,
        rotation_angles=ROTATION_ANGLES,
        crop_size=CROP_SIZE,
        max_precomputed_islands=MAX_PRECOMPUTED_ISLANDS,
        nr_sigmas=NR_SIGMAS,
        rms=RMS,
        stretch_type=STRETCH_TYPE,
        segmentation_mode=SEGMENTATION_MODE,
        save_dir=DATASET_SAVE_DIR
    ).build()
    logger.info(f"Finished building the dataset.")

    elapsed_time = timeit.default_timer() - start_time
    logger.info(f"Dataset generation pipeline completed successfully in {elapsed_time:.2f} seconds.")


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
