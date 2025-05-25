import json
import os
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Process collision scenario data for VLM input.")
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the input JSON file"
)
args = parser.parse_args()

# Load the input JSON file
with open(args.input_path, "r") as f:
    datas = json.load(f)

# Define the fixed prompt for the user without specifying type
USER_PROMPT_NO_TYPE = (
    "You are a collision scenario analysis expert. Based on the traffic scenario described in the input images, "
    "your task is to identify the vehicle most likely to generate collision with the ego vehicle. "
    "The scene consists of six camera views surrounding the ego vehicle, arranged as follows: "
    "The first row includes three images: FRONT LEFT, FRONT, and FRONT RIGHT. "
    "The second row includes three images: BACK RIGHT, BACK, and BACK LEFT. "
    "Potential Dangerous Vehicles are highlighted with red boxes, and each vehicle's ID is labeled in the top-left corner of the respective box. "
    "Select the one most likely to have its future trajectory modified (through manual intervention) to produce the collision with the ego vehicle. "
    "The speed of any car other than ego vehicle can be adjusted, as long as it is in accordance with the laws of physics, so there is no need to analyze the speed of other cars. "
    "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. In the current scenario, the initial speed of the ego vehicle is "
)

# List to hold processed samples
no_type_new_datas = []

# Set to track images already processed (to avoid duplication)
image_seen_set = set()

# Iterate over all input data samples
for i in range(len(datas)):
    image_name = datas[i]["rgb_path"].split("/")[-1]

    # Skip duplicate images
    if image_name in image_seen_set:
        continue
    image_seen_set.add(image_name)

    # Create new dictionary for each sample
    new_data = {
        "id": i,
        "image": image_name,
        "conversations": [],
        "full_image_path": datas[i]["rgb_path"],
        "sample_token": datas[i]["sample_token"],
        "ego_vehicle_speed": datas[i]["ego_init_speed"],
        "token": datas[i]["token"]
    }

    # Add the user prompt
    question = {
        "from": "human",
        "value": USER_PROMPT_NO_TYPE + "%.1fm/s" % datas[i]["ego_init_speed"] + "."
    }
    new_data["conversations"].append(question)

    # If collision_dict exists, append it as an answer
    if "collision_dict" in datas[i]:
        collision_dict = datas[i]['collision_dict']
        keys = [k for k, v in collision_dict.items() if v != 0]
        value_dict = {k: collision_dict[k] for k in keys}
        answer = {
            "from": "human",
            "value": value_dict
        }
        new_data["conversations"].append(answer)
        new_data["collision_dict"] = datas[i]["collision_dict"]

    # Append the processed sample to the output list
    no_type_new_datas.append(new_data)

# Build the output file path by appending "_VLM_input.json" to the input file name
input_dir = os.path.dirname(args.input_path)
input_name = os.path.splitext(os.path.basename(args.input_path))[0]
output_path = os.path.join(input_dir, input_name + "_VLM_input.json")

# Write the processed data to the output JSON file
with open(output_path, "w") as f:
    json.dump(no_type_new_datas, f, indent=4)
