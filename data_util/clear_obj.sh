#!/bin/bash

input_folders=("processed_cups" "water_bottle_processed" "beer_bottle")

# delete partA_new.obj, partB_new.obj

for folder in "${input_folders[@]}"; do
  for subfolder in $(find "$folder" -type d); do
    partA_obj="$subfolder/partA_new.obj"
    partB_obj="$subfolder/partB_new.obj"
    # partA_ply="$subfolder/partA-pc.ply"
    # partB_ply="$subfolder/partB-pc.ply" 

    if [[ -f "$partA_obj" && -f "$partB_obj" ]]; then
      echo "$partA_obj and $partB_obj found, deleted"
      rm "$partA_obj"
      rm "$partB_obj"
    else
      echo "partA_new.obj and partB_new.obj is messing"
    fi
  done
done