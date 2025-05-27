#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
# timestamp=$(date +"%Y%m%d_%H%M%S")
OUTPUT_NAME="250_val_random3"
OUTPUT_DIR="../../experiment"
DATA_COUNT=250
RANDOM_SEED=3
SPLIT_TYPE="val"
RAW_INPUT_JSON_PATH="${OUTPUT_DIR}/annotation/${OUTPUT_NAME}.json"
VLM_INPUT_JSON_PATH="${OUTPUT_DIR}/annotation/${OUTPUT_NAME}_VLM_input.json"
VLM_INFERENCE_JSON_PATH="${OUTPUT_DIR}/annotation/${OUTPUT_NAME}_VLM_inference.json"
IMAGE_DIR="${OUTPUT_DIR}/example_rgb_250_val_random3"
NUSCENES_ROOT="../../nuscenes"

# Run the first script to generate the input
echo "Running data generation..."
python ./tools/generate/generate_input_for_VLM_for_val.py \
  --output_name ${OUTPUT_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --data_count ${DATA_COUNT} \
  --random_seed ${RANDOM_SEED} \
  --split_type ${SPLIT_TYPE} \
  --dataroot ${NUSCENES_ROOT}

# Run the second script to convert the JSON and add additional keys
echo "Running JSON conversion and augmentation..."
python ./tools/convert/json_convert_add_ego_V_add_noType_collide_question_add_additional_key.py \
  --input_path ${RAW_INPUT_JSON_PATH}

# VLM select
echo "Running VLM selection..."
python ./inference/inference.py --model_base_dir ../../weights/vlm-selector --data_path ${VLM_INPUT_JSON_PATH} --image_root ${IMAGE_DIR} --output_path ${VLM_INFERENCE_JSON_PATH}

echo "Done."