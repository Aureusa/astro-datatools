#!/usr/bin/env bash

# Time the transfer process
START_TIME=$(date +%s)

read -r -p "Paste the dataset location on strw.maas: " DATASET_LOCATION
DATASET_FOLDER=$(basename "$DATASET_LOCATION")

echo "Sending '$DATASET_LOCATION' to alice..."
sleep 3

SSH_KEY="$HOME/.ssh/alice3"

rsync -avz --progress \
  -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes" \
  "$DATASET_LOCATION" \
  "alice3:/home/s4861264/project_data/$DATASET_FOLDER"

echo "Transfer successful!"

# Calculate and display the total time taken for the transfer
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "Total time taken for the transfer: $TOTAL_TIME seconds"
