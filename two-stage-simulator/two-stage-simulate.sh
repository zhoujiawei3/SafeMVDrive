#!/bin/bash

python scripts/scene_editor_json_input_test_crash_or_VLM_reason.py \
    --results_root_dir ../experiment/trajectory_simulation/collision_stage_250_val_random3_simulation/ \
    --num_scenes_per_batch 1 \
    --env trajdata \
    --dataset_path nuscenes_trainval_1.0 \
    --policy_ckpt_dir ../weights/two-stage-simulator \
    --policy_ckpt_key iter80000.ckpt \
    --eval_class Diffuser \
    --editing_source 'config' 'heuristic' \
    --registered_name 'trajdata_nusc_diff_based_current_false' \
    --render \
    --simulation_json_path ../experiment/annotation/250_val_random3_VLM_inference.json \
    --cuda 0 \
    --refix collision_simulation \
    --trajectory_output


python scripts/scene_editor_json_input_second_time_simulation.py \
    --results_root_dir ../experiment/trajectory_simulation/evasion_stage_250_val_random3_simulation/  \
    --num_scenes_per_batch 1 \
    --env=trajdata \
    --dataset_path nuscenes_trainval_1.0 \
    --policy_ckpt_dir ../weights/two-stage-simulator \
    --policy_ckpt_key iter80000.ckpt \
    --eval_class Diffuser \
    --editing_source 'config' 'heuristic' \
    --registered_name trajdata_nusc_diff_based_current_false \
    --render \
    --trajdata_source_test val \
    --simulation_json_path ../experiment/annotation/250_val_random3_VLM_inference_collision_simulation-trajectory.json \
    --cuda 0 \
    --trajectory_output \
    --trajrefix evasion_simulation_trajectory 
