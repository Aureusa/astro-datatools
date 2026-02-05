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

# Append astro-datatools path for imports
sys.path.append('/home/penchev/astro-datatools/')
from astro_datatools import setup_logging

logger = logging.getLogger("coco_dataset_splitter")


def split_coco_dataset_by_components(
        coco_json_path: str,
        output_dir: str,
        splits: dict = None,
        seed: int = 42
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
    
    # Validate splits sum to 1.0
    total = sum(splits.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    np.random.seed(seed)
    
    # Load COCO annotations
    logger.info(f"Loading COCO dataset from {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get the source directory for images and proposals
    source_dir = Path(coco_json_path).parent
    images_dir = source_dir / "images"
    proposals_dir = source_dir / "proposals"
    
    # Group images by component count
    component_count_groups = defaultdict(list)
    image_id_to_data = {}
    
    # First pass: build image_id_to_data for ALL images
    for img in coco_data['images']:
        image_id_to_data[img['id']] = img
    
    # Second pass: filter and group by component count (only non-rotated origin images)
    for img in tqdm(
        coco_data['images'],
        desc="Processing images"
    ):
        # Check if both bbox and segm annotations exist for this image
        has_bbox = any(ann['image_id'] == img['id'] and 'bbox' in ann for ann in coco_data['annotations'])
        has_segm = any(ann['image_id'] == img['id'] and 'segmentation' in ann for ann in coco_data['annotations'])
        if not (has_bbox and has_segm):
            logger.warning(f"Image ID {img['id']} is missing bbox or segmentation annotations.")
            continue

        # Skip rotated images - we only want to split based on origin images
        if 'metadata' in img and 'rotated' in img['metadata']:
            if img['metadata']['rotated']:
                continue

        # Extract component count from metadata
        if 'metadata' in img and 'grg_positions' in img['metadata']:
            component_count = len(img['metadata'].get('grg_positions', []))
            if component_count == 0:
                logger.warning(f"Image ID {img['id']} has zero GRG positions in metadata.")
                continue
        
        component_count_groups[component_count].append(img['id'])
    
    # Perform stratified split for each component count group
    split_image_ids = {split_name: [] for split_name in splits.keys()}
    split_names = list(splits.keys())
    split_ratios = [splits[name] for name in split_names]
    
    logger.info("Stratified splitting by component count:")
    component_stats = {}
    
    for component_count, image_ids in tqdm(
        sorted(component_count_groups.items()), 
        desc="Splitting by component count"
    ):
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
        
        # Track statistics
        component_stats[component_count] = {
            split_name: len(split_ids) 
            for split_name, split_ids in zip(split_names, image_id_splits)
        }
        component_stats[component_count]['total'] = n_images
        
        split_info = " | ".join([f"{name}: {len(ids)}" for name, ids in zip(split_names, image_id_splits)])
        logger.info(f"Component count {component_count}: {n_images} images -> {split_info}")
    
    # Create COCO datasets for each split and copy files
    output_stats = {}
    
    for split_name, image_ids in tqdm(
        split_image_ids.items(),
        desc="Creating split datasets"
    ):
        # Expand image_ids to include all rotated versions of the origin images
        expanded_image_ids = set(image_ids)
        
        for origin_id in image_ids:
            # Find all images that have this origin_id in their metadata
            for img in coco_data['images']:
                if 'metadata' in img and 'origin_id' in img['metadata']:
                    if img['metadata']['origin_id'] == origin_id:
                        expanded_image_ids.add(img['id'])
        
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
        
        for img_id in expanded_image_ids:
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
        for ann in coco_data['annotations']:
            if ann['image_id'] in image_ids_set:
                split_coco['annotations'].append(ann)
        
        # Save split annotations
        split_json_path = split_dir / "annotations.json"
        with open(split_json_path, 'w') as f:
            json.dump(split_coco, f, indent=4)
        
        logger.info(f"Saved {split_name} annotations to {split_json_path}")
        logger.info(f"Images: {len(split_coco['images'])}, Annotations: {len(split_coco['annotations'])}")
        
        # Track statistics
        output_stats[split_name] = {
            'num_images': len(split_coco['images']),
            'num_annotations': len(split_coco['annotations']),
            'json_path': str(split_json_path)
        }
    
    # Print summary statistics
    logger.info("="*60)
    logger.info("SPLIT SUMMARY")
    logger.info("="*60)
    
    for split_name in split_names:
        stats = output_stats[split_name]
        logger.info(f"{split_name.upper()}:")
        logger.info(f"  Images: {stats['num_images']}")
        logger.info(f"  Annotations: {stats['num_annotations']}")
        logger.info(f"  Location: {stats['json_path']}")
    
    logger.info("="*60)
    logger.info("COMPONENT COUNT DISTRIBUTION")
    logger.info("="*60)
    
    for component_count in sorted(component_stats.keys()):
        stats = component_stats[component_count]
        logger.info(f"Component count {component_count} ({stats['total']} total):")
        for split_name in split_names:
            count = stats[split_name]
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            logger.info(f"  {split_name}: {count} ({percentage:.1f}%)")
    
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
    
    # Save component count distribution
    component_distribution_data = []
    for component_count in sorted(component_stats.keys()):
        stats = component_stats[component_count]
        row = {
            'component_count': component_count,
            'total': stats['total']
        }
        for split_name in split_names:
            count = stats[split_name]
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            row[f'{split_name}_count'] = count
            row[f'{split_name}_percentage'] = round(percentage, 1)
        component_distribution_data.append(row)
    
    component_distribution_df = pd.DataFrame(component_distribution_data)
    component_distribution_csv = output_dir_path / "component_distribution.csv"
    component_distribution_df.to_csv(component_distribution_csv, index=False)
    logger.info(f"Saved component distribution to {component_distribution_csv}")
    
    return {
        'output_stats': output_stats,
        'component_stats': component_stats
    }


if __name__ == "__main__":
    start_time = timeit.default_timer()
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split COCO dataset by component count with stratification."
    )
    parser.add_argument(
        "coco_json",
        type=str,
        help="Path to the COCO annotations JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where split datasets will be saved"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()

    # Setup logging
    log_filepath = os.path.join(args.output_dir, "dataset_pipeline.log")
    setup_logging(log_file=log_filepath, level="DEBUG")
    
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
        'train': args.train,
        'val': args.val,
        'test': args.test
    }
    
    split_coco_dataset_by_components(
        coco_json_path=args.coco_json,
        output_dir=args.output_dir,
        splits=splits,
        seed=args.seed
    )
    elapsed_time = timeit.default_timer() - start_time
    logger.info(f"Dataset splitting completed successfully in {elapsed_time:.2f} seconds.")