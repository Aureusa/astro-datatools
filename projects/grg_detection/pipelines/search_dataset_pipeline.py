from astropy.table import Table
import sys
import timeit
import os
import logging
from tqdm import tqdm
import yaml
import argparse

# Append astro-datatools path for imports
sys.path.append('/home/penchev/astro-datatools/')
from astro_datatools import setup_logging

# Append project path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from grg_detection.coco import GRGSearchDatasetBuilder

# Append strw_lofar_data_utils path for imports
sys.path.append('/home/penchev/strw_lofar_data_utils/')
from src.core.mosaic_crawler import DR2Crawler

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

def main(config_path: str):
    start_time = timeit.default_timer()
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration values for PATHS
    COMPONENT_CATALOGUE_FILEPATH = config['PATHS']['COMPONENT_CATALOGUE_FILEPATH']
    DATASET_SAVE_DIR = config['PATHS']['SAVE_DIR']
    
    # Extract configuration values for CUTOUT_PARAMS
    CUTOUT_SIZE = config['CUTOUT_PARAMS']['CUTOUT_SIZE']
    RMS = config['CUTOUT_PARAMS']['RMS']
    NR_SIGMAS = config['CUTOUT_PARAMS']['NR_SIGMAS']
    
    # Extract configuration values for AUGMENTATION
    STRETCH_TYPE = config['AUGMENTATION']['STRETCH_TYPE']
    MAX_PRECOMPUTED_ISLANDS = config['AUGMENTATION']['MAX_PRECOMPUTED_ISLANDS']
    
    # Extract configuration values for MOSAIC_TO_CRAWL
    MOSAICS_TO_CRAWL = config['MOSAICS_TO_CRAWL']['MOSAICS_NAME']
    STRIDE = config['MOSAICS_TO_CRAWL']['STRIDE']

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
    
    # # Crawl the specified mosaics and generate cutouts
    # cutouts = []
    # for mosaic in tqdm(MOSAICS_TO_CRAWL, desc="Crawling mosaics"):
    #     crawler = DR2Crawler(mosaic, CUTOUT_SIZE, STRIDE, verbose=False)
    #     results = crawler.crawl()
    #     cutouts.extend(results)

    cutouts = DR2Crawler('P219+32', CUTOUT_SIZE, STRIDE, verbose=False).crawl()  # For testing with a single mosaic

    logger.info(f"Finished crawling mosaics. Total cutouts generated: {len(cutouts)}")
    logger.info("Building dataset with all cutouts...")
    GRGSearchDatasetBuilder(
        cutouts=cutouts,
        component_catalogue=COMPONENT_CATALOGUE,
        max_precomputed_islands=MAX_PRECOMPUTED_ISLANDS,
        nr_sigmas=NR_SIGMAS,
        rms=RMS,
        stretch_type=STRETCH_TYPE,
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
