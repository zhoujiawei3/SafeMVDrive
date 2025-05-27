#!/bin/bash
trajectory_file_path="../experiment/annotation/250_val_random3_VLM_inference_collision_simulation-trajectory_evasion_simulation_trajectory.json"
dataset_type="adversarial_trajectory"
bash generate_collide_dataset.sh $trajectory_file_path  $dataset_type 


PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --nproc_per_node="4" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="12356" \
  src/dwm/export_generation_result_as_nuscenes_data.py \
  -c configs/unimlvg/generate_result_as_nuscenes_data.json \
  -o ../experiment/video_nuscenes_4_30_250_val_random3

cp -r ../nuscenes/v1.0-collision ../experiment/video_nuscenes_4_30_250_val_random3/
cp -r ../nuscenes/maps ../experiment/video_nuscenes_4_30_250_val_random3/
# We use interpolation to generate the ego status of the ego vehicle. The duplication here is solely for compatibility with the evaluation code in UniAD.
cp -r ../nuscenes/can_bus ../experiment/video_nuscenes_4_30_250_val_random3/ 