from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from Levenshtein import ratio
# steps = 500
# MODEL_PATH=f"/data/shz/project/llama-factory/LLaMA-Factory/saves/qwen2_5_vl-3b/full/sft/checkpoint-{steps}" 
# OUTPUT_PATH="./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_sft_{STEPS}.json"

MODEL_PATH = "/data/MLLM_models/models--Qwen--Qwen2.5-VL-72B-Instruct"

# MODEL_PATH= "/data3/zhoujiawei/finetune/VLM-R1/100_random-bev-linearLR-hasbeta/checkpoint-100"
OUTPUT_PATH = "./logs_5_15/no_type_question/rgb_results_250_val_random3_bug_fix_before20frames_72B_baseline.json"

BSZ=1
DATA_ROOT = "/home/zhoujiawei/VLM_attack_propose/annotation/mini-data_new_250_val_random3_bug_fix_before_20_frames_auto_label_test_false_current_speed_add_ego_V_add_noType_collide_question_only_no_type.json"


IMAGE_ROOT = "/home/zhoujiawei/VLM_attack_propose/example_rgb_250_val_random3_bug_fix_before_20_frames"

# cuda_device="cuda:1"

random.seed(42)

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    # device_map=cuda_device
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_bbox_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_match =re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match[0].strip() if content_match else content.strip()
    # if content_answer_match:
    return student_answer
    # else:
    #     return None

sample_data=250
data = json.load(open(DATA_ROOT, "r"))
examples=[]
for x in data:
    new_example = {}
    image_path = os.path.join(IMAGE_ROOT, x['image'])
    new_example['image'] = image_path
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": x['conversations'][0]['value'] +'Output final answer (number) in <answer> </answer> tags.'
                }
            ]
        }]
    new_example['message'] = message
    new_example['ground_truth'] = x['conversations'][1]['value']
    # new_example["collision_type"]=x['conversations'][0]['value'].split("The desired generated collision type is ")[1].split(".")[0]
    if "collision_dict" in x.keys():
        new_example["collision_dict"]=x["collision_dict"]
    examples.append(new_example)
examples=examples[:sample_data]
for i in tqdm(range(0, len(examples), BSZ)):
    batch_examples = examples[i:i + BSZ]
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
    inputs = inputs.to("cuda")
    # inputs = inputs.to(cuda_device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    for j, output in enumerate(batch_output_text):
        examples[i+j]['output'] = output

proposed_car_number=0
collided_car_number=0
print_result=[]
print(len(examples))
for example in examples:
    judge=0
    output_extract = extract_bbox_answer(example['output'])
    if not(output_extract=="" or output_extract==None):
        #判断是否是数字
        if output_extract.isdigit():
            proposed_car_number+=1
            if output_extract in example['collision_dict']:
                if example['collision_dict'][output_extract]==1:
                    collided_car_number+=1
        

    result = {
        'image_path' : example['image'],
        'ground_truth': example['ground_truth'],
        'model_output': example['output'],
        'model_answer': output_extract,
        'judge': judge
    }
    print_result.append(result)
epison=1e-6


print("----------------------------------------")
print("OUTPUT_PATH",OUTPUT_PATH)


print("proposed_car_number",proposed_car_number)
print("collided_car_number",collided_car_number)

output_path = OUTPUT_PATH
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_path, 'w') as f:
    json.dump({
            'proposed_car_number': proposed_car_number,
            'collided_car_number': collided_car_number,
            'results': print_result
        }, f, indent=2)

print(f"Results saved to {output_path}")
print("-"*100)
torch.cuda.empty_cache()
# for ds in TEST_DATASETS:
#     print(f"Processing {ds}...")
#     ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
#     data = json.load(open(ds_path, "r"))
#     random.shuffle(data)
#     QUESTION_TEMPLATE = "{Question}"
#     data = data[:sample_num]
#     messages = []

#     for x in data:
#         image_path = os.path.join(IMAGE_ROOT, x['image'])
#         message = [
#             # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#             {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image", 
#                     "image": f"file://{image_path}"
#                 },
#                 {
#                     "type": "text",
#                     "text": QUESTION_TEMPLATE.format(Question=x['problem'])
#                 }
#             ]
#         }]
#         messages.append(message)

#     all_outputs = []  # List to store all answers

#     # Process data
#     for i in tqdm(range(0, len(messages), BSZ)):
#         batch_messages = messages[i:i + BSZ]
    
#         # Preparation for inference
#         text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
#         image_inputs, video_inputs = process_vision_info(batch_messages)
#         inputs = processor(
#             text=text,
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             padding_side="left",
#             return_tensors="pt",
#         )
#         inputs = inputs.to("cuda:0")

#         # Inference: Generation of the output
#         generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         batch_output_text = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
        
#         all_outputs.extend(batch_output_text)
#         # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

#     final_output = []
#     correct_number = 0

#     for input_example, model_output in zip(data, all_outputs):
#         original_output = model_output
#         ground_truth = input_example['solution']
#         ground_truth_normalized = input_example['normalized_solution']
#         model_answer, normalized = extract_bbox_answer(original_output)
        
#         # Count correct answers
#         correct = 0
#         if model_answer is not None:
#             if not normalized and iou(model_answer, ground_truth) > 0.5:
#                 correct = 1
#             elif normalized and iou(model_answer, ground_truth_normalized) > 0.5:
#                 correct = 1
#         correct_number += correct
        
#         # Create a result dictionary for this example
#         result = {
#             'question': input_example['problem'],
#             'ground_truth': ground_truth,
#             'model_output': original_output,
#             'extracted_answer': model_answer,
#             'correct': correct
#         }
#         final_output.append(result)

#     # Calculate and print accuracy
#     accuracy = correct_number / len(data) * 100
#     print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

#     # Save results to a JSON file
#     output_path = OUTPUT_PATH.format(DATASET=ds)
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     with open(output_path, "w") as f:
#         json.dump({
#             'accuracy': accuracy,
#             'results': final_output
#         }, f, indent=2)

#     print(f"Results saved to {output_path}")
#     print("-"*100)





