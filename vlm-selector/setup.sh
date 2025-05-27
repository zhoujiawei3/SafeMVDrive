# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1
# Install the packages in open-r1-multimodal .
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils
pip install flash-attn --no-build-isolation
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]
pip install nuscenes-devkit
pip install -U "numpy<2.0.0"
pip install peft==0.14.0