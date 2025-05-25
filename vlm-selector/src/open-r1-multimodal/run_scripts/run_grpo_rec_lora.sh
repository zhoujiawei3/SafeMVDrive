cd /home/zhoujiawei/VLM_attack_propose/VLM-R1/src/open-r1-multimodal

export DEBUG_MODE="true"

RUN_NAME="1500_bev_continue_train_after6frames_random7_only_no_type-consine_7B_lora_64_128_0.05_lr_2e-5_deepseed_3"
export LOG_PATH="/data3/zhoujiawei/logs/debug_log_$RUN_NAME.txt"
export WANDB_DIR="/data3/zhoujiawei/"
# --model_name_or_path /data3/zhoujiawei/finetune/VLM-R1/SFT_7B_FULL_nothingFreeze_manual_50_labelWithCOT_no_chinese/checkpoint-300 \

CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12396" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir /data3/zhoujiawei/finetune/VLM-R1/$RUN_NAME \
    --model_name_or_path /data/MLLM_models/models--Qwen--Qwen2.5-VL-7B-Instruct \
    --deepspeed local_scripts/zero3.json \
    --data_file_paths /home/zhoujiawei/VLM_attack_propose/annotation/mini-data_new_1500_bev_continue_train_after6frames_random7_only_no_type.json \
    --dataset_name car \
    --image_folders /home/zhoujiawei/SafeMVDrive/experiment/20250525_185935/example_rgb_250_val_random3 \
    --max_completion_length 1024 \
    --num_generations 6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 20 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules false \
    --lr_scheduler_type cosine \
    --save_only_model false \
