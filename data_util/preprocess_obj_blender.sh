#!/bin/bash

input_folders=("processed_cups" "water_bottle_processed" "beer_bottle")

for folder in "${input_folders[@]}"; do
  for subfolder in $(find "$folder" -type d); do
    partA_obj="$subfolder/partA.obj"
    partB_obj="$subfolder/partB.obj"
    # partA_ply="$subfolder/partA-pc.ply"
    # partB_ply="$subfolder/partB-pc.ply" 

    if [[ -f "$partA_obj" && -f "$partB_obj" ]]; then
      echo "Processing $subfolder"
      python preprocess_obj_blender.py "$subfolder"
      # python csv_to_ply.py --input_csv "$partB_csv" --output_ply "$partB_ply"
    else
      echo "Skipping $subfolder: partA.obj or partB.obj not found"
    fi
  done
done