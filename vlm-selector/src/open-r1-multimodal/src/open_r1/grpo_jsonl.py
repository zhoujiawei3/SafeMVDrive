# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
# from torch.optim import AdamW
# #导入常数优化器 
# from torch.optim.lr_scheduler import LambdaLR


# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging
import json
from datasets import Dataset
logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    vlm_trainer: Optional[str] = field(
        default="default",
        metadata={
            "help": "Choose VLM trainer type: 'default', 'modified', 'modified_bf16', or 'modified_optimized_bf16'",
            "choices": ["default", "modified", "modified_bf16", "modified_optimized_bf16"]
        },
    )
    
       

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    # print(completions[0][0]["content"])
    # contents = [completion[0]["content"].split("assistant\n")[1] for completion in completions]
    contents = [completion[0]["content"] for completion in completions]
    # print(type(completions[0][0]["content"].))
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):

        sol=json.loads(sol)
        image_path=sol.pop("image_path")
        reward = 0.0
        try:
            content_match = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
            # print(f"student_answer: {content}\n")
            # print(f"content_match[0]: {content_match[0].strip()}\n")
            # print(f"content_match[1]: {content_match[1].strip()}\n")
            student_answer = content_match[0].strip() if content_match else content.strip()
            # print(f"student_answer: {student_answer}\n")
            if not sol:
                reward = ratio(student_answer.lower(), "no vehicle is appropriate".lower())
            else:
                if student_answer in sol.keys():
                    reward = sol[str(student_answer)]
                else:
                    reward = 0
           
        except Exception:
            pass 

        # Try symbolic verification first for numeric answers
        # try:
        #     answer = parse(content)
        #     if float(verify(answer, parse(sol))) > 0:
        #         reward = 1.0
        # except Exception:
        #     pass  # Continue to next verification method if this fails

        # # If symbolic verification failed, try string matching or fuzzy matching
        # if reward == 0.0:
        #     try:
        #         # Extract answer from solution if it has think/answer tags
        #         sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        #         ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
        #         # Extract answer from content if it has think/answer tags
        #         content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        #         student_answer = content_match.group(1).strip() if content_match else content.strip()
                
        #         # Check if ground truth contains any numbers
        #         has_numbers = bool(re.search(r'\d', ground_truth))
                
        #         if has_numbers:
        #             # For numeric answers, use exact matching
        #             reward = 1.0 if student_answer == ground_truth else 0.0
        #         else:
        #             # For text answers, use fuzzy matching
        #             reward = ratio(student_answer.lower(), ground_truth.lower())
                
        #     except Exception:
        #         pass  # Keep reward as 0.0 if all methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"image_path: {image_path}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # for content, match in zip(completion_contents, matches):
    #     if not bool(match):
    #         print(f"Content: {content}\n")
    #         print(f"Has format: {bool(match)}\n")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # if os.getenv("DEBUG_MODE") == "true":
    #     log_path = os.getenv("LOG_PATH")
    #     with open(log_path, "a", encoding='utf-8') as f:
    #         f.write(f"------------- {current_time} Format reward -------------\n")
    #         for content, match in zip(completion_contents, matches):
    #             f.write(f"Content: {content}\n")
    #             f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
     freeze_vision_modules: bool = False

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            data= json.load(f)

            for item in data:
                new_item = {}
                # Store image path instead of loading the image
                new_item['image_path'] = os.path.join(image_folder, item['image'])
                # Remove immediate image loading
                new_item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                #把image_path塞入solution用于debug
                item['conversations'][1]['value']['image_path'] = new_item['image_path']
                new_item['solution'] = json.dumps(item['conversations'][1]['value'])
                # print(f"new_item['solution']:{new_item['solution']}")
                del item['image'] # remove the image column so that it can be loaded later
                # print(new_item)
                all_data.append(new_item)
    
    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        # Don't load image here, just store the path
        return {
            'image_path': example['image_path'],  # Store path instead of loaded image
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'text': None},
                    {'type': 'text', 'text': example['problem'] + '  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.'}
                ]
            }]
        }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)
    
    # Split dataset for validation if requested
    splits = {'train': dataset}
    print(f"split ratio: {script_args.val_split_ratio}")
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    
    trainer_cls = Qwen2VLGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    print(f"eval strategy:{training_args.eval_strategy}")
    # Initialize the GRPO trainer

    
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits['validation'] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels
    )

    # Train and push the model to the Hub
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    # parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    # print(parser)
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    main(script_args, training_args, model_args)
