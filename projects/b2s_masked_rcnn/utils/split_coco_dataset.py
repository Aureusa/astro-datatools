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


def _resolve_category_ids(coco_data: dict) -> tuple[int, int]:
    """Return the dataset category ids for SCS and MCS."""
    scs_id = None
    mcs_id = None

    for category in coco_data.get('categories', []):
        name = str(category.get('name', '')).strip().lower()
        if name == 'scs':
            scs_id = category['id']
        elif name == 'mcs':
            mcs_id = category['id']

    if scs_id is None:
        scs_id = 1
    if mcs_id is None:
        mcs_id = 2

    return scs_id, mcs_id


def _is_rotated_image(image: dict) -> bool:
    metadata = image.get('metadata', {})
    return bool(metadata.get('rotated', False))


def _build_image_type_groups(
        coco_data: dict,
        annotations_by_image: dict,
        scs_category_id: int,
        mcs_category_id: int,
        logger: logging.Logger
    ) -> tuple[defaultdict, dict]:
    """Group non-rotated origin images by the source types present in their annotations."""
    images_with_scs = set()
    images_with_mcs = set()

    for ann in tqdm(coco_data['annotations'], desc="Indexing annotation source types"):
        if ann['category_id'] == scs_category_id:
            images_with_scs.add(ann['image_id'])
        elif ann['category_id'] == mcs_category_id:
            images_with_mcs.add(ann['image_id'])

    image_type_groups = defaultdict(list)
    image_type_by_id = {}

    for img in tqdm(coco_data['images'], desc="Grouping origin images by source type"):
        if _is_rotated_image(img):
            continue

        image_id = img['id']
        has_scs = image_id in images_with_scs
        has_mcs = image_id in images_with_mcs

        if has_scs and has_mcs:
            image_type = 'both'
        elif has_scs:
            image_type = 'scs_only'
        elif has_mcs:
            image_type = 'mcs_only'
        else:
            image_type = 'unlabeled'

        image_type_groups[image_type].append(image_id)
        image_type_by_id[image_id] = image_type

    logger.info(
        "Origin image type counts before splitting: "
        f"SCS only={len(image_type_groups['scs_only'])}, "
        f"MCS only={len(image_type_groups['mcs_only'])}, "
        f"Both={len(image_type_groups['both'])}, "
        f"Unlabeled={len(image_type_groups['unlabeled'])}"
    )

    return image_type_groups, image_type_by_id


def _count_image_types(
        image_ids: list[int],
        annotations_by_image: dict,
        scs_category_id: int,
        mcs_category_id: int
    ) -> dict:
    """Count how many images contain SCS, MCS, both, or neither."""
    counts = {
        'total': len(image_ids),
        'scs_images': 0,
        'mcs_images': 0,
        'both_images': 0,
        'unlabeled_images': 0,
    }

    for image_id in image_ids:
        category_ids = {ann['category_id'] for ann in annotations_by_image.get(image_id, [])}
        has_scs = scs_category_id in category_ids
        has_mcs = mcs_category_id in category_ids

        if has_scs:
            counts['scs_images'] += 1
        if has_mcs:
            counts['mcs_images'] += 1
        if has_scs and has_mcs:
            counts['both_images'] += 1
        if not has_scs and not has_mcs:
            counts['unlabeled_images'] += 1

    return counts


def split_coco_dataset_by_components(
        coco_data: dict,
        coco_json_path: str,
        output_dir: str,
        splits: dict = None,
        seed: int = 42,
        logger: logging.Logger = None,
        copy_assets: bool = True,
    ) -> dict:
    """
    Split a COCO dataset into train/val/test sets with stratification by source type.

    Non-rotated origin images are grouped by the annotation categories they contain:
    SCS only, MCS only, both, or neither. The split is performed on those origin
    images, then all rotated variants are expanded into the same split as their origin.
    The function can optionally copy image and proposal files to the appropriate
    split directories, or save split annotations only.
    
    :param coco_json_path: Path to the COCO annotations JSON file.
    :type coco_json_path: str
    :param output_dir: Directory where split datasets will be saved.
    :type output_dir: str
    :param splits: Dictionary with split ratios, e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}.
                   Defaults to {'train': 0.7, 'val': 0.15, 'test': 0.15}.
    :type splits: dict
    :param seed: Random seed for reproducibility.
    :type seed: int
    :param copy_assets: If True, copy image/proposal files into split folders.
                        If False, only save split annotations JSON.
    :type copy_assets: bool
    :return: Dictionary with statistics about the splits.
    :rtype: dict
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if splits is None:
        splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    np.random.seed(seed)
    
    # Get the source directory for images and proposals
    source_dir = Path(coco_json_path).parent
    images_dir = source_dir / "images"
    proposals_dir = source_dir / "proposals"
    
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

    scs_category_id, mcs_category_id = _resolve_category_ids(coco_data)
    logger.info(
        f"Using category ids for split stratification: SCS={scs_category_id}, MCS={mcs_category_id}"
    )
    logger.info(f"Copy assets mode: {copy_assets}")

    image_type_groups, image_type_by_id = _build_image_type_groups(
        coco_data=coco_data,
        annotations_by_image=annotations_by_image,
        scs_category_id=scs_category_id,
        mcs_category_id=mcs_category_id,
        logger=logger,
    )
    
    # Perform stratified split for each image type group
    split_image_ids = {split_name: [] for split_name in splits.keys()}
    split_names = list(splits.keys())
    split_ratios = [splits[name] for name in split_names]
    
    logger.info("Assigning images to splits...")
    for image_type, image_ids in image_type_groups.items():
        if not image_ids:
            logger.info(f"Skipping empty image type group: {image_type}")
            continue

        image_ids = np.array(image_ids)
        np.random.shuffle(image_ids)
        
        n_images = len(image_ids)
        
        # Calculate split indices
        cumulative_ratios = np.cumsum(split_ratios)
        split_indices = [int(n_images * ratio) for ratio in cumulative_ratios[:-1]]
        
        # Split the image IDs
        image_id_splits = np.split(image_ids, split_indices)
        
        logger.info(
            f"Assigning {n_images} origin images from group '{image_type}' across splits {splits}"
        )

        for split_name, split_ids in zip(split_names, image_id_splits):
            split_image_ids[split_name].extend(split_ids.tolist())

    for split_name, image_ids in split_image_ids.items():
        origin_type_counts = {
            'scs_only': 0,
            'mcs_only': 0,
            'both': 0,
            'unlabeled': 0,
        }
        for image_id in image_ids:
            origin_type_counts[image_type_by_id[image_id]] += 1

        logger.info(
            f"Origin split '{split_name}': total={len(image_ids)}, "
            f"SCS only={origin_type_counts['scs_only']}, "
            f"MCS only={origin_type_counts['mcs_only']}, "
            f"Both={origin_type_counts['both']}, "
            f"Unlabeled={origin_type_counts['unlabeled']}"
        )
    
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

        origin_label_counts = _count_image_types(
            image_ids=image_ids,
            annotations_by_image=annotations_by_image,
            scs_category_id=scs_category_id,
            mcs_category_id=mcs_category_id,
        )
        expanded_label_counts = _count_image_types(
            image_ids=expanded_image_ids,
            annotations_by_image=annotations_by_image,
            scs_category_id=scs_category_id,
            mcs_category_id=mcs_category_id,
        )

        logger.info(
            f"Split '{split_name}' origin image labels: "
            f"SCS={origin_label_counts['scs_images']}, "
            f"MCS={origin_label_counts['mcs_images']}, "
            f"Both={origin_label_counts['both_images']}, "
            f"Unlabeled={origin_label_counts['unlabeled_images']}"
        )
        logger.info(
            f"Split '{split_name}' total image labels including rotations: "
            f"SCS={expanded_label_counts['scs_images']}, "
            f"MCS={expanded_label_counts['mcs_images']}, "
            f"Both={expanded_label_counts['both_images']}, "
            f"Unlabeled={expanded_label_counts['unlabeled_images']}"
        )
        
        # Create split directories
        split_dir = Path(output_dir) / split_name
        split_images_dir = split_dir / "images"
        split_proposals_dir = split_dir / "proposals"

        # Always create the split directory because annotations.json is written there.
        split_dir.mkdir(parents=True, exist_ok=True)
        
        if copy_assets:
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
            
            if copy_assets:
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
            'json_path': str(split_json_path),
            'origin_label_counts': origin_label_counts,
            'expanded_label_counts': expanded_label_counts,
        }
    
    # Print summary statistics
    logger.info("SPLIT SUMMARY:")
    for split_name in split_names:
        stats = output_stats[split_name]
        origin_label_counts = stats['origin_label_counts']
        expanded_label_counts = stats['expanded_label_counts']
        info = (
            f"{split_name.upper()} saved to {stats['json_path']}:\n"
            f"- {stats['num_images']} images\n"
            f"- {stats['num_annotations']} annotations\n"
            f"- origin labels: SCS={origin_label_counts['scs_images']}, "
            f"MCS={origin_label_counts['mcs_images']}, "
            f"Both={origin_label_counts['both_images']}, "
            f"Unlabeled={origin_label_counts['unlabeled_images']}\n"
            f"- total labels incl. rotations: SCS={expanded_label_counts['scs_images']}, "
            f"MCS={expanded_label_counts['mcs_images']}, "
            f"Both={expanded_label_counts['both_images']}, "
            f"Unlabeled={expanded_label_counts['unlabeled_images']}"
        )
        logger.info(info)

    # Save statistics to CSV files
    output_dir_path = Path(output_dir)
    
    # Save split summary
    split_summary_data = []
    for split_name in split_names:
        stats = output_stats[split_name]
        origin_label_counts = stats['origin_label_counts']
        expanded_label_counts = stats['expanded_label_counts']
        split_summary_data.append({
            'split': split_name,
            'num_images': stats['num_images'],
            'num_annotations': stats['num_annotations'],
            'json_path': stats['json_path'],
            'origin_scs_images': origin_label_counts['scs_images'],
            'origin_mcs_images': origin_label_counts['mcs_images'],
            'origin_both_images': origin_label_counts['both_images'],
            'origin_unlabeled_images': origin_label_counts['unlabeled_images'],
            'total_scs_images': expanded_label_counts['scs_images'],
            'total_mcs_images': expanded_label_counts['mcs_images'],
            'total_both_images': expanded_label_counts['both_images'],
            'total_unlabeled_images': expanded_label_counts['unlabeled_images'],
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
        description="Split COCO dataset by source type while keeping rotated variants together."
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
    copy_assets_input = input("Copy image/proposal files into split folders? (y/n, default y): ").strip().lower()
    copy_assets = copy_assets_input != 'n'

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
    print(f"Copy assets: {copy_assets}")

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
        logger=logger,
        copy_assets=copy_assets,
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
