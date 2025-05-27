#!/bin/bash
trajectory_file_path=$1
dataset_type=$2
echo ">>> Script execution started..."

set -e
echo ">>> Strict mode enabled (exit on command failure)."

echo ">>> Removing old directories..."
rm -rf ./text_description/collide
rm -rf ../nuscenes/interp_12Hz_trainval_collide
rm -rf ../nuscenes/v1.0-collision
rm -f ../nuscenes/v1.0-trainval-zip/interp_12Hz_trainval_collide.zip
rm -rf ASAP/out
rm -rf ../nuscenes/new_samples
rm -rf ../nuscenes/new_sweeps
rm -rf ../nuscenes/v1.0-trainval-zip/new_sweeps.zip
rm -rf ../nuscenes/v1.0-trainval-zip/new_samples.zip
echo ">>> Old directories removed."
 

# Run nusc_annotation_changer.py script
echo ">>> Running nusc_annotation_changer.py script..."
python tools/nusc_annotation_changer_new.py --trajectory-file=${trajectory_file_path} --dataset_type=${dataset_type} 
echo ">>> nusc_annotation_changer.py script completed."

# Switch to ASAP directory and activate ASAP environment
cd ASAP
echo ">>> Switched to ASAP directory."

# Run ann_generator.sh script
echo ">>> Running ann_generator.sh script..."
bash scripts/ann_generator.sh 12 --ann_strategy 'interp'
echo ">>> ann_generator.sh script completed."

# Switch to nuscenes data directory
cd ../../nuscenes
echo ">>> Switched to nuscenes data directory."

# Compress interp_12Hz_trainval_collide directory
echo ">>> Compressing interp_12Hz_trainval_collide directory..."
zip -r interp_12Hz_trainval_collide.zip interp_12Hz_trainval_collide/
echo ">>> Compression completed: interp_12Hz_trainval_collide.zip"

# Move the zip file to target directory
echo ">>> Moving compressed file to v1.0-trainval-zip directory..."
mv interp_12Hz_trainval_collide.zip v1.0-trainval-zip
echo ">>> File moved to target directory."

# Compress new_samples directory
echo ">>> Compressing new_samples directory..."
zip -r new_samples.zip new_samples/
echo ">>> Compression completed: new_samples"

# Move the zip file to target directory
echo ">>> Moving compressed file to v1.0-trainval-zip directory..."
mv new_samples.zip v1.0-trainval-zip
echo ">>> File moved to target directory."

# Compress new_sweeps directory
echo ">>> Compressing new_sweeps directory..."
zip -r new_sweeps.zip new_sweeps/
echo ">>> Compression completed: new_sweeps"

# Move the zip file to target directory
echo ">>> Moving compressed file to v1.0-trainval-zip directory..."
mv new_sweeps.zip v1.0-trainval-zip
echo ">>> File moved to target directory."


# Final message
echo ">>> Script finished!"