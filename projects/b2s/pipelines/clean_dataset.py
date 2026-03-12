import logging
import json

from grg_detection.coco.clean import COCODatasetCleaner
from grg_detection.coco.evaluator import GTEvaluator

from astro_datatools.logger import setup_logging

logger = setup_logging(name="grg_detection.pipelines.clean_dataset_pipeline")


def main(filepath_for_coco: str):
    # Clean the dataset using COCODatasetCleaner
    logger.info("Cleaning the COCO dataset...")
    
    # Load the COCO dataset from the provided filepath (should be json)
    with open(filepath_for_coco, 'r') as f:
        coco = json.load(f)

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean the COCO dataset and evaluate against ground truth.")
    parser.add_argument("--annotations-filepath", type=str, help="Path to the COCO annotations file.", default="/net/vdesk/data2/penchev/project_data/big-dataset-class-ratio-5/annotations.json")
    args = parser.parse_args()
    main(args.annotations_filepath)
