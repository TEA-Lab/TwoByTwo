#!/bin/bash

input_folders=("processed_cups" "water_bottle_processed" "beer_bottle")

for folder in "${input_folders[@]}"; do
  for subfolder in $(find "$folder" -type d); do
    partA_csv="$subfolder/partA-pc.csv"
    partB_csv="$subfolder/partB-pc.csv"
    partA_ply="$subfolder/partA-pc.ply"
    partB_ply="$subfolder/partB-pc.ply" 

    if [[ -f "$partA_csv" && -f "$partB_csv" ]]; then
      echo "Processing $subfolder"
      python csv_to_ply.py --input_csv "$partA_csv" --output_ply "$partA_ply"
      python csv_to_ply.py --input_csv "$partB_csv" --output_ply "$partB_ply"
    else
      echo "Skipping $subfolder: partA.obj or partB.obj not found"
    fi
  done
done