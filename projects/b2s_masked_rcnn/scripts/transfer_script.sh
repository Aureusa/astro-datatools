#!/usr/bin/env bash

read -r -p "Paste the dataset location on strw.maas: " DATASET_LOCATION
DATASET_FOLDER=$(basename "$DATASET_LOCATION")

ALICE_DIR="/home/s4861264/project_data/$DATASET_FOLDER"

echo "Destination on ALICE: $ALICE_DIR"\

echo "Sending '$DATASET_LOCATION' to ALICE..."
sleep 3

SSH_KEY="$HOME/.ssh/alice3"

# Time the transfer process
START_TIME=$(date +%s)

rsync -avz --progress \
  -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes" \
  "$DATASET_LOCATION" \
  "alice3:$ALICE_DIR"

echo "Transfer successful!"

# Calculate and display the total time taken for the transfer
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Convert total time to a more human-readable format
if [ $TOTAL_TIME -lt 60 ]; then
  echo "Total time taken for the transfer: $TOTAL_TIME seconds"
elif [ $TOTAL_TIME -lt 3600 ]; then
  MINUTES=$((TOTAL_TIME / 60))
  SECONDS=$((TOTAL_TIME % 60))
  echo "Total time taken for the transfer: $MINUTES minutes and $SECONDS seconds"
else
  HOURS=$((TOTAL_TIME / 3600))
  REMAINDER=$((TOTAL_TIME % 3600))
  MINUTES=$((REMAINDER / 60))
  SECONDS=$((REMAINDER % 60))
  echo "Total time taken for the transfer: $HOURS hours, $MINUTES minutes and $SECONDS seconds"
fi
