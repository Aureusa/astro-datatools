#!/bin/bash
# -----------------------------------------------------------------------------------
# Script to create training, validation, and test datasets for GRG detection
# -----------------------------------------------------------------------------------
PATH_TO_PROJECT=/home/penchev/astro-datatools/projects/grg_detection # Path to the GRG project on the system
DATASET_CONFIG_PATH=$PATH_TO_PROJECT/configs/dataset_pipeline.yaml # Path to the dataset pipeline config file
OUTPUT_FOLDER="train-val-test-split" # Folder name for the train-val-test split within the output directory
ANNOTATIONS_FILE="annotations.json" # Name of the annotations file in COCO format
# -----------------------------------------------------------------------------------

# Get the output directory for the dataset from the config file
DATASET_OUTPUT_DIR=$(python3 -c "import yaml; config = yaml.safe_load(open('$DATASET_CONFIG_PATH')); print(config['PATHS']['SAVE_DIR'])")

# Get the last part of the output directory path (the folder name) and print it
DATASET_OUTPUT_FOLDER=$(basename $DATASET_OUTPUT_DIR)

echo "Output directory for dataset: $DATASET_OUTPUT_DIR"

# Create COCO dataset
python3 $PATH_TO_PROJECT/pipelines/dataset_pipeline.py --config $DATASET_CONFIG_PATH

# Split dataset into training, validation, and test sets
python3 $PATH_TO_PROJECT/utils/split_coco_dataset.py $DATASET_OUTPUT_DIR/$ANNOTATIONS_FILE --output-dir $DATASET_OUTPUT_DIR/$OUTPUT_FOLDER --train 0.7 --val 0.15 --test 0.15
# -----------------------------------------------------------------------------------