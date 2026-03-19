import json
import sys
from tqdm import tqdm
import timeit
import numpy as np
from collections import defaultdict
from pathlib import Path
import shutil
import logging
import os
import pandas as pd

from astro_datatools.logger import setup_logging


def split_coco_dataset_by_components(
        coco_data: dict,
        coco_json_path: str,
        output_dir: str,
        splits: dict = None,
        seed: int = 42,
        logger: logging.Logger = None
    ) -> dict:
    """
    Split a COCO dataset into train/val/test sets with stratification by component count.
    
    This ensures that GRGs with different numbers of components are evenly distributed
    across all splits. Also handles copying image and proposal files to the appropriate
    split directories.
    
    :param coco_json_path: Path to the COCO annotations JSON file.
    :type coco_json_path: str
    :param output_dir: Directory where split datasets will be saved.
    :type output_dir: str
    :param splits: Dictionary with split ratios, e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}.
                   Defaults to {'train': 0.7, 'val': 0.15, 'test': 0.15}.
    :type splits: dict
    :param seed: Random seed for reproducibility.
    :type seed: int
    :return: Dictionary with statistics about the splits.
    :rtype: dict
    """
    if splits is None:
        splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    np.random.seed(seed)
    
    # Get the source directory for images and proposals
    source_dir = Path(coco_json_path).parent
    images_dir = source_dir / "images"
    proposals_dir = source_dir / "proposals"
    
    # Group images by component count
    component_count_groups = defaultdict(list)
    image_id_to_data = {}
    annotations_by_image = defaultdict(list)
    origin_to_augmented_ids = defaultdict(list)
    
    # First pass: build image_id_to_data and origin->augmented index for ALL images
    for img in coco_data['images']:
        image_id_to_data[img['id']] = img
        metadata = img.get('metadata', {})
        origin_id = metadata.get('origin_id')
        if origin_id is not None:
            origin_to_augmented_ids[origin_id].append(img['id'])

    # Build annotation indexes once
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        annotations_by_image[image_id].append(ann)
    
    # Second pass: filter and group by component count (only non-rotated origin images)
    for img in tqdm(
        coco_data['images'],
        desc="Processing images"
    ):
        # Skip rotated images - we only want to split based on origin images
        if 'metadata' in img and 'rotated' in img['metadata']:
            if img['metadata']['rotated']:
                continue
        component_count_groups[1].append(img['id'])
    
    # Perform stratified split for each component count group
    split_image_ids = {split_name: [] for split_name in splits.keys()}
    split_names = list(splits.keys())
    split_ratios = [splits[name] for name in split_names]
    
    logger.info("Assigning images to splits...")
    for component_count, image_ids in component_count_groups.items():
        # Shuffle the image IDs for this component count
        image_ids = np.array(image_ids)
        np.random.shuffle(image_ids)
        
        n_images = len(image_ids)
        
        # Calculate split indices
        cumulative_ratios = np.cumsum(split_ratios)
        split_indices = [int(n_images * ratio) for ratio in cumulative_ratios[:-1]]
        
        # Split the image IDs
        image_id_splits = np.split(image_ids, split_indices)
        
        # Assign to splits
        for split_name, split_ids in zip(split_names, image_id_splits):
            split_image_ids[split_name].extend(split_ids.tolist())
    
    # Create COCO datasets for each split and copy files
    output_stats = {}
    
    for split_name, image_ids in split_image_ids.items():
        # Expand image_ids to include all rotated versions of the origin images
        expanded_image_ids = set(image_ids)
        
        for origin_id in tqdm(image_ids, desc=f"Expanding {split_name} images"):
            # Add all augmented images that belong to this origin image
            expanded_image_ids.update(origin_to_augmented_ids.get(origin_id, []))
        
        expanded_image_ids = list(expanded_image_ids)
        logger.info(f"Creating {split_name} dataset with {len(image_ids)} origin images "
                   f"({len(expanded_image_ids)} total including rotations)...")
        
        # Create split directories
        split_dir = Path(output_dir) / split_name
        split_images_dir = split_dir / "images"
        split_proposals_dir = split_dir / "proposals"
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_proposals_dir.mkdir(parents=True, exist_ok=True)
        
        # Create COCO structure for this split
        split_coco = {
            'images': [],
            'annotations': [],
            'categories': coco_data['categories']
        }
        
        # Get all annotations for these images (including rotated versions)
        image_ids_set = set(expanded_image_ids)
        
        for img_id in tqdm(expanded_image_ids, desc=f"Processing {split_name} images"):
            img_data = image_id_to_data[img_id]
            split_coco['images'].append(img_data)
            
            # Copy image file
            img_filename = img_data['file_name']
            src_img = images_dir / img_filename
            dst_img = split_images_dir / img_filename
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Copy proposal file (if exists)
            proposal_filename = img_filename.replace('.png', '.npz')
            src_proposal = proposals_dir / proposal_filename
            dst_proposal = split_proposals_dir / proposal_filename
            
            if src_proposal.exists():
                shutil.copy2(src_proposal, dst_proposal)
        
        # Get annotations for these images
        for img_id in tqdm(image_ids_set, desc=f"Processing {split_name} annotations"):
            split_coco['annotations'].extend(annotations_by_image.get(img_id, []))
        
        # Save split annotations
        split_json_path = split_dir / "annotations.json"
        with open(split_json_path, 'w') as f:
            json.dump(split_coco, f, indent=4)
        
        logger.info(f"Saved {split_name} annotations to {split_json_path}. Images: {len(split_coco['images'])}, Annotations: {len(split_coco['annotations'])}")
        
        # Track statistics
        output_stats[split_name] = {
            'num_images': len(split_coco['images']),
            'num_annotations': len(split_coco['annotations']),
            'json_path': str(split_json_path)
        }
    
    # Print summary statistics
    logger.info("SPLIT SUMMARY:")
    for split_name in split_names:
        stats = output_stats[split_name]
        info = f"{split_name.upper()} saved to {stats['json_path']}:\n- {stats['num_images']} images\n- {stats['num_annotations']} annotations"
        logger.info(info)

    # Save statistics to CSV files
    output_dir_path = Path(output_dir)
    
    # Save split summary
    split_summary_data = []
    for split_name in split_names:
        stats = output_stats[split_name]
        split_summary_data.append({
            'split': split_name,
            'num_images': stats['num_images'],
            'num_annotations': stats['num_annotations'],
            'json_path': stats['json_path']
        })
    
    split_summary_df = pd.DataFrame(split_summary_data)
    split_summary_csv = output_dir_path / "split_summary.csv"
    split_summary_df.to_csv(split_summary_csv, index=False)
    logger.info(f"Saved split summary to {split_summary_csv}")
    
    return {
        'output_stats': output_stats,
    }


if __name__ == "__main__":
    start_time = timeit.default_timer()
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split COCO dataset by component count with stratification."
    )
    # parser.add_argument(
    #     "coco_json",
    #     type=str,
    #     help="Path to the COCO annotations JSON file"
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     type=str,
    #     required=True,
    #     help="Directory where split datasets will be saved"
    # )
    # parser.add_argument(
    #     "--train",
    #     type=float,
    #     default=0.7,
    #     help="Train split ratio (default: 0.7)"
    # )
    # parser.add_argument(
    #     "--val",
    #     type=float,
    #     default=0.15,
    #     help="Validation split ratio (default: 0.15)"
    # )
    # parser.add_argument(
    #     "--test",
    #     type=float,
    #     default=0.15,
    #     help="Test split ratio (default: 0.15)"
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=42,
    #     help="Random seed for reproducibility (default: 42)"
    # )
    
    # args = parser.parse_args()

    exists = False
    while not exists:
        input_prompt = "Enter path to COCO annotations JSON file (or 'exit' to quit): "
        coco_filepath = input(input_prompt)
        if coco_filepath.lower() == 'exit':
            print("Aborting.")
            sys.exit(0)

        # Check if the file exists
        if not os.path.isfile(coco_filepath):
            print(f"Error: File '{coco_filepath}' does not exist.")
            input_prompt = "Please enter a valid path to COCO annotations JSON file (or 'exit' to quit): "
            exists = False
        else:
            exists = True    

    output_dir = input("Enter output directory for split datasets: ")

    train_ratio = float(input("Enter train split ratio (default 0.7): ") or 0.7)
    val_ratio = float(input("Enter validation split ratio (default 0.15): ") or 0.15)
    test_ratio = float(input("Enter test split ratio (default 0.15): ") or 0.15)

    valid_ratio = False
    while not valid_ratio:
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            print(f"Error: Split ratios must sum to 1.0, got {total}. Please re-enter the ratios.")
            train_ratio = float(input("Enter train split ratio (default 0.7): ") or 0.7)
            val_ratio = float(input("Enter validation split ratio (default 0.15): ") or 0.15)
            test_ratio = float(input("Enter test split ratio (default 0.15): ") or 0.15)
        else:
            valid_ratio = True

    seed = int(input("Enter random seed for reproducibility (default 42): ") or 42)
    
    with open(coco_filepath, 'r') as f:
        coco_data = json.load(f)

    nr_images = len(coco_data['images'])
    nr_annotations = len(coco_data['annotations'])

    # Overview of the configuration
    print("\nConfiguration:")
    print(f"COCO JSON file: {coco_filepath}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {nr_images} (annotations {nr_annotations})")
    print(f"Train split ratio: {train_ratio} ({int(train_ratio*nr_images)} images)")
    print(f"Validation split ratio: {val_ratio} ({int(val_ratio*nr_images)} images)")
    print(f"Test split ratio: {test_ratio} ({int(test_ratio*nr_images)} images)")
    print(f"Random seed: {seed}")

    continue_confirm = input("Continue with these settings? (y/n): ")
    if continue_confirm.lower() != 'y':
        print("Aborting.")
        sys.exit(0)

    # Setup logging
    log_filepath = os.path.join(output_dir, "dataset_pipeline.log")
    logger = setup_logging(name="b2s.pipelines.split_coco_dataset", log_file=log_filepath)
    
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
    
    splits = {
        'train': train_ratio,
        'val': val_ratio,
        'test': test_ratio
    }
    
    split_coco_dataset_by_components(
        coco_data=coco_data,
        coco_json_path=coco_filepath,
        output_dir=output_dir,
        splits=splits,
        seed=seed,
        logger=logger
    )
    elapsed_time = timeit.default_timer() - start_time
    if elapsed_time < 60: # less than a minute
        logger.info(f"Dataset splitting completed successfully in in {elapsed_time:.2f} seconds.")
    elif elapsed_time < 3600: # less than an hour
        minutes, seconds = divmod(elapsed_time, 60)
        logger.info(f"Dataset splitting completed successfully in in {int(minutes)} minutes and {seconds:.2f} seconds.")
    else:
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Dataset splitting completed successfully in in {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds.")
