#!/bin/bash

input_folders=("processed_cups" "water_bottle_processed" "beer_bottle")

for folder in "${input_folders[@]}"; do
  for subfolder in $(find "$folder" -type d); do
    partA_file="$subfolder/partA_new.obj"
    partB_file="$subfolder/partB_new.obj"

    if [[ -f "$partA_file" && -f "$partB_file" ]]; then
      echo "Processing $subfolder"
      ./generate_pc "$partA_file" "$partB_file" "$subfolder"
    else
      echo "Skipping $subfolder: partA_new.obj or partB_new.obj not found"
    fi
  done
done
