import json
json_path="/home/zhoujiawei/VLM_attack_propose/annotation/mini-data_new_250_val_random3_bug_fix_before20frames_VLM_no_type_inference_trajectory_ablation_study_alpha_1_beta1_50_gama1_1_beta2_50_gama2_1_lambda_9e-1.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
total_distance=0.0
for item in data:
    distance = item["min_distance"]
    total_distance+=distance
avg_distance= total_distance/float(len(data))
print("escape_scene_number:",len(data))
print("avg_distance:",avg_distance)
