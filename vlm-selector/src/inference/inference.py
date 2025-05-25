from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import re
import os
from tqdm import tqdm
import argparse
from peft import PeftModel, PeftConfig

def extract_bbox_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_match = re.findall(answer_tag_pattern, content, re.DOTALL)
    student_answer = content_match[0].strip() if content_match else content.strip()
    return student_answer

def extract_collision_type(text):
    try:
        # Try to extract the collision type from the format in your example
        collision_part = text.split("The desired generated collision type is ")
        if len(collision_part) > 1:
            collision_type = collision_part[1].split(".")[0].strip()
            return collision_type
        return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Run inference with Qwen2.5-VL model')
    parser.add_argument('--model_name', type=str, default="", help='Path to the model checkpoint')
    parser.add_argument('--model_base_dir', type=str, help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, default='/home/zhoujiawei/VLM_attack_propose/annotation/mini-data_new_500_continue_val_random3_bug_fix_after6frames_false_current_speed_add_ego_V_add_noType_collide_question_adjustPrompt_only_no_type.json', help='Path to the input JSON data')
    parser.add_argument('--image_root', type=str, default="/home/zhoujiawei/VLM_attack_propose/example_rgb_500_continue_val_random3_bug_fix_before_20_frames", help='Root directory for images')
    parser.add_argument('--output_path', type=str, default='/home/zhoujiawei/VLM_attack_propose/annotation/mini-data_new_500_continue_val_random3_bug_fix_before20frames_VLM_no_type_inference.json', help='Path to save the output JSON')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run inference on')
    parser.add_argument('--step', type=int, default=2600, help='Step for finetuned model')
    parser.add_argument('--max_samples', type=int, default=-1, help='Maximum number of samples to process, -1 for all')
    
    args = parser.parse_args()
    
    # Load model and processor
    model_name=args.model_name
    model_base_dir=args.model_base_dir
    steps = args.step
    if model_name!='':
        model_path = f"{model_base_dir}/{model_name}/checkpoint-{steps}"
    else:
        model_path = f"{model_base_dir}/checkpoint-{steps}"
    print(f"Loading model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Check if LoRA weights exist and load them if they do
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading LoRA weights")
        processor = AutoProcessor.from_pretrained(model_path)
        lora_config = PeftConfig.from_pretrained(model_path)  # 替换为您的LoRA权重路径        
        model = PeftModel.from_pretrained(model, model_path)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    # Limit samples if specified
    if args.max_samples > 0:
        data = data[:args.max_samples]
    
    print(f"Processing {len(data)} examples")
    
    # Prepare examples
    examples = []
    for x in data:
        image_path = os.path.join(args.image_root, x['image'])
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        if len(x['conversations']) < 2:
            ground_truth = ""
        else:
            ground_truth = x['conversations'][1]['value']
        new_example = {
            'image': image_path,
            'sample_token': x.get('sample_token', ''),
            'message': [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{image_path}"
                        },
                        {
                            "type": "text",
                            "text": x['conversations'][0]['value'] + '  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.'
                        }
                    ]
                }
            ],
            'ground_truth': ground_truth,
            'question': x['conversations'][0]['value'],
        }
        
        # Only add collision_type if it exists in the input
        collision_type = extract_collision_type(x['conversations'][0]['value'])
        if collision_type:
            new_example['collision_type'] = collision_type
            
        # Include collision_dict if it exists in the input
        if "collision_dict" in x:
            new_example["collision_dict"] = x["collision_dict"]
        if 'ego_vehicle_speed' in x:
            new_example['ego_vehicle_speed'] = x['ego_vehicle_speed']
        if 'token' in x:
            new_example['token'] = x['token']
        examples.append(new_example)
    
    # Run inference
    results = []
    for i in tqdm(range(0, len(examples), args.batch_size)):
        batch_examples = examples[i:i + args.batch_size]
        
        # Process data
        batch_messages = [x['message'] for x in batch_examples]
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(args.device)
        
        # Generate responses
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Process results
        for j, output in enumerate(batch_output_text):
            example = batch_examples[j]
            answer_extract = extract_bbox_answer(output)
            
            result = {
                'sample_token': example['sample_token'],
                'image_path': example['image'],
                'question': example['question'],
                'full_response': output,
                'answer': answer_extract,
                'ground_truth': example['ground_truth']
            }
            
            # Only include collision_type if it exists in the example
            if 'collision_type' in example:
                result['collision_type'] = example['collision_type']
            if 'ego_vehicle_speed' in example:
                result['ego_init_v'] = example['ego_vehicle_speed']
            if 'token' in example:
                result['token'] = example['token']
            
            results.append(result)
    
    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference completed. Processed {len(results)} examples.")

if __name__ == "__main__":
    main()