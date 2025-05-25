from nuscenes import NuScenes
import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
import sys
import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import BoxVisibility
from PIL import ImageDraw, ImageFont
import random
from nuscenes.utils.splits import create_splits_scenes
import pickle
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box
import copy
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='NuScenes Data Generator with Configurable Parameters')
    parser.add_argument('--random_seed', type=int, default=3, help='Random seed for reproducibility')
    parser.add_argument('--data_count', type=int, default=250, help='Number of data samples to generate')
    parser.add_argument('--dataroot', type=str, default='nuscenes', 
                       help='Root directory of NuScenes dataset')
    parser.add_argument('--output_name', type=str, default='250_val_random3', 
                       help='Output dataset name suffix')
    parser.add_argument('--output_dir', type=str, default='./experiment',
                       help='Output directory for generated data')
    parser.add_argument('--split_type', type=str, choices=['train', 'val'], default='val',
                       help='Which dataset split to use (train or val)')
    return parser.parse_args()

def get_ego_speed_from_sample(nusc, sample_token):
    """
    Calculate ego vehicle speed based on sample token and previous LiDAR frame data
    
    Args:
        nusc: NuScenes instance
        sample_token: Current sample token
        
    Returns:
        speed: Ego vehicle speed in m/s
    """
    # Get current sample
    sample = nusc.get('sample', sample_token)
    
    # Get current frame LiDAR data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    
    # Get ego_pose corresponding to LiDAR
    current_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    current_position = np.array(current_ego_pose['translation'])
    current_timestamp = current_ego_pose['timestamp'] / 1000000  # Convert to seconds
    
    # Check if there is previous LiDAR frame data
    prev_lidar_token = lidar_data['prev']
    if not prev_lidar_token:
        # If current LiDAR has no previous frame, check if there is a previous sample
        prev_sample_token = sample['prev']
        if not prev_sample_token:
            print("This is the first frame of the scene with no previous LiDAR data, cannot calculate speed")
            return 0.0
        
        # Get previous sample's LiDAR data
        prev_sample = nusc.get('sample', prev_sample_token)
        prev_lidar_token = prev_sample['data']['LIDAR_TOP']
    
    # Get previous frame LiDAR data
    prev_lidar_data = nusc.get('sample_data', prev_lidar_token)
    prev_ego_pose = nusc.get('ego_pose', prev_lidar_data['ego_pose_token'])
    prev_position = np.array(prev_ego_pose['translation'])
    prev_timestamp = prev_ego_pose['timestamp'] / 1000000  # Convert to seconds
    
    # Calculate time difference (seconds)
    dt = current_timestamp - prev_timestamp
    if dt <= 0:
        print("Timestamp anomaly, cannot calculate speed")
        return 0.0
    
    # Calculate position difference (meters)
    displacement = current_position - prev_position
    
    # Calculate horizontal displacement distance (ignore z-axis)
    distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
    
    # Calculate speed (meters/second)
    speed = distance / dt
    
    return speed

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def get_2d_center_in_camera(annotation_token, nusc, camera_name='CAM_FRONT'):
    """
    Given the annotation token, this function calculates the 2D projection of the object center in the specified camera view.
    
    :param annotation_token: Token for a sample annotation.
    :param nusc: The NuScenes dataset object.
    :param camera_name: Camera name (default: 'CAM_FRONT')
    
    :return: The 2D (x, y) coordinates of the object center in the specified camera view.
    """
    # Retrieve the annotation record
    ann_rec = nusc.get('sample_annotation', annotation_token)
    
    # Get the token for the associated sample_data
    sample_token = ann_rec['sample_token']
    sample_data = nusc.get('sample', sample_token)['data']
    
    cam_front_token = sample_data[camera_name]
    
    # Get camera intrinsic parameters
    sd_rec = nusc.get('sample_data', cam_front_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    
    # Get the annotation box (3D bounding box)
    box = nusc.get_box(ann_rec['token'])
    translation = ann_rec['translation']
    
    # Retrieve ego pose (translation and rotation)
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    translation_ego = np.array(pose_rec['translation'])
    rotation_ego = Quaternion(pose_rec['rotation'])
    
    # Retrieve calibrated sensor pose (translation and rotation)
    translation_sensor = np.array(cs_rec['translation'])
    rotation_sensor = Quaternion(cs_rec['rotation'])
    
    # Step 1: Transform the 3D box center from the world (ego) coordinate system to the sensor coordinate system
    box.translate(-translation_ego)  # Translate by ego translation
    box.rotate(rotation_ego.inverse)  # Rotate by inverse ego rotation
    
    box.translate(-translation_sensor)  # Translate by sensor translation
    box.rotate(rotation_sensor.inverse)  # Rotate by inverse sensor rotation
    
    # Step 2: Get the 3D corners of the box and take the center point
    corners_3d = box.corners()  # 3D corners of the bounding box
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    corners_3d = corners_3d[:, in_front]
    
    center_3d = box.center  # The center of the 3D bounding box
    
    # Step 3: Project the 3D center to 2D using camera intrinsic parameters
    corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()  # Project 3D corners to 2D
    final_coords = post_process_coords(corner_coords)
    
    if final_coords is None:
        return None, None, None, None
    else:
        x_min, y_min, x_max, y_max = final_coords
    
    # Get the center point of the new rectangle
    x_box, y_box = (x_min + x_max) // 2, (y_min + y_max) // 2
    height = y_max - y_min
    width = x_max - x_min
    
    return (int(x_box), int(y_box), int(height), int(width))


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set parameters from arguments
    dataroot = args.dataroot
    data_count = args.data_count
    random_seed = args.random_seed
    output_name = args.output_name
    output_dir = args.output_dir
    split_type = args.split_type
    
    # Determine dataset version based on dataroot
    output_dataset = dataroot.split('/')[-1]
    version = 'v1.0-mini' if 'mini' in dataroot else 'v1.0-trainval'
    
    # Initialize NuScenes
    nuscenes = NuScenes(version, dataroot=dataroot)
    predict_helper = PredictHelper(nuscenes)

    # Create scene splits
    scene_splits = create_splits_scenes()
    print("Scene splits:", scene_splits)
    
    train_split = scene_splits['train']
    val_split = scene_splits['val']
    train_list = []
    val_list = []
    
    for i, scene in enumerate(nuscenes.scene):
        print(f"Processing scene {i}")
        if scene['name'] in train_split:
            train_list.append(i)
        elif scene['name'] in val_split:
            val_list.append(i)

    # Choose which split to use based on argument
    if split_type == 'train':
        selected_list = train_list
        print(f"Using training split with {len(train_list)} scenes")
    else:
        selected_list = val_list
        print(f"Using validation split with {len(val_list)} scenes")

    # Set random seed
    random.seed(random_seed)

    # Set up paths
    data_save_path = os.path.join(output_dir, f"annotation/{output_name}.json")
    rgb_output_dir = os.path.join(output_dir, f"example_rgb_{output_name}")
    rgb_temp_dir = os.path.join(output_dir, 'rgb_image')

    # Create directories
    os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(rgb_temp_dir, exist_ok=True)

    collision_type_list = [
        'A vehicle cuts in and collides with the ego vehicle',
        'A vehicle rear-ends the ego vehicle',
        'Ego vehicle rear-ends another vehicle',
        'A vehicle has a head-on collision with the ego vehicle',
        'A vehicle has a T-bone collision with the ego vehicle'
    ]
    annotations = []

    while len(annotations) < data_count * 5:
        meta_annotation = {
            'sample_token': None,
            'rgb_path': None,
            'collision': '',
            'ego_init_speed': 0,
            'reward': {},
            'token': {}
        }
        
        output_scene_number = random.choice(selected_list)
        scene = nuscenes.scene[output_scene_number]
        output_sample_number = random.choice(range(1, scene['nbr_samples']-20))

        sample_token = scene['first_sample_token']
        count = output_sample_number
        for i in range(count):
            sample_token = nuscenes.get('sample', sample_token)['next']
        
        meta_annotation['sample_token'] = sample_token
        meta_annotation['ego_init_speed'] = get_ego_speed_from_sample(nuscenes, sample_token)
        
        # Check if already in annotations
        already_in_annotation = False
        for annotation in annotations:
            if annotation['sample_token'] == sample_token:
                already_in_annotation = True
                break
        if already_in_annotation:
            continue

        # Process annotations
        annos_tokens = nuscenes.get('sample', sample_token)['anns']
        lidar_sample_data = nuscenes.get('sample_data', nuscenes.get('sample', sample_token)['data']['LIDAR_TOP'])
        sd_ep = nuscenes.get("ego_pose", lidar_sample_data["ego_pose_token"])
        ego_translation = sd_ep['translation']

        anno_need_to_render_dict = {}
        annos_tokens = sorted(annos_tokens, key=lambda anno_token: np.max(
            np.linalg.norm(np.array(ego_translation) - np.array(nuscenes.get('sample_annotation', anno_token)['translation']))))
        
        count = 0
        for anno_token in annos_tokens:
            anno = nuscenes.get('sample_annotation', anno_token)
            if not (anno['category_name'].split('.')[0] == 'vehicle' and 
                   anno['category_name'].split('.')[1] != 'bicycle' and 
                   anno['category_name'].split('.')[1] != 'motorcycle'):
                continue
            
            distance = np.linalg.norm(np.array(ego_translation) - np.array(anno['translation']))
            visibility_token = anno['visibility_token']
            
            if distance > 25:
                continue
            count = count + 1
            if distance > 5 and visibility_token != '4':
                continue
            anno_need_to_render_dict[str(count)] = anno_token
            print(f"Distance: {distance}")

        

        # Process RGB images
        view_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
                    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        file_list = []
        
        for i, view in enumerate(view_list):
            sample_data_token = nuscenes.get('sample', sample_token)['data'][view]
            sample_data = nuscenes.get('sample_data', sample_data_token)
            # Save image to rgb_temp_dir
            shutil.copy(os.path.join(dataroot, sample_data['filename']), 
                       os.path.join(rgb_temp_dir, view + ".jpg"))
            file_list.append(os.path.join(rgb_temp_dir, view + ".jpg"))

        images = {}
        for i, file in enumerate(file_list):
            image = Image.open(file)
            # Get 2D box
            draw = ImageDraw.Draw(image)

            for label, anno in anno_need_to_render_dict.items():
                x_box, y_box, height, width = get_2d_center_in_camera(anno, nuscenes, view_list[i])
                
                if x_box is None:
                    continue
                
                # Draw the box
                image_width, image_height = image.size
                x_center_pixel = x_box
                y_center_pixel = y_box
                box_width_pixel = width
                box_height_pixel = height

                # Calculate rectangle corners
                x1 = x_center_pixel - box_width_pixel / 2
                y1 = y_center_pixel - box_height_pixel / 2
                x2 = x_center_pixel + box_width_pixel / 2
                y2 = y_center_pixel + box_height_pixel / 2

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                
                # Draw label
                label_text = f"{label}"
                font = ImageFont.load_default(70)

                if y1 - 30 >= 0:
                    # Use textbbox to calculate text bounding box
                    text_bbox = draw.textbbox((x1, y1 - 30), label_text, font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Draw label background box
                    draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
                    
                    # Draw label text
                    draw.text((x1, y1-text_height-18), label_text, fill="white", font=font)
                else:
                    # Use textbbox to calculate text bounding box
                    text_bbox = draw.textbbox((x1, y1 + 30), label_text, font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    # Draw label background box
                    draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill="red")

                    # Draw label text
                    draw.text((x1, y1-18), label_text, fill="white", font=font)

            # Resize to 896*448
            image = image.resize((896, 448))
            images[i] = image

        # Store reward and token info
        for label, anno in anno_need_to_render_dict.items():
            meta_annotation['reward'][f"{label}"] = 0
            meta_annotation['token'][f"{label}"] = nuscenes.get('sample_annotation', anno)["instance_token"]
        
        # Create combined RGB image
        border_colors = [
            (0, 255, 255),  # Cyan
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 165, 0),  # Orange
            (255, 105, 180) # Hot Pink
        ]
        
        width, height = images[0].size
        new_image = Image.new('RGB', (width * 3 + 12, height * 2 + 8), (255, 255, 255))
        
        for i in range(2):
            for j in range(3):
                img = images[i * 3 + j]
                # Add border to image
                border_color = border_colors[i * 3 + j]
                bordered_img = Image.new('RGB', (width + 4, height + 4), border_color)
                bordered_img.paste(img, (2, 2))

                # Paste bordered image to new image
                new_image.paste(bordered_img, (j * (width + 4), i * (height + 4)))

                # Add text in top-left corner
                draw = ImageDraw.Draw(new_image)
                font_size = 36
                font = ImageFont.load_default(font_size)
                
                text = f'{view_list[i * 3 + j]}'
                draw.text((j * (width + 4) + 10, i * (height + 4) + 10), text, font=font, fill=(255, 255, 255))

        # Save images
        rgb_filename = f"{output_dataset}_{output_scene_number}_{output_sample_number}.jpg"
        
        new_image.save(os.path.join(rgb_output_dir, rgb_filename))

        # Set paths in annotation
        meta_annotation['rgb_path'] = os.path.join(rgb_output_dir, rgb_filename)

        # Create annotations for each collision type
        for collision_type in collision_type_list:
            new_meta_annotation = copy.deepcopy(meta_annotation)
            new_meta_annotation['collision'] = collision_type
            annotations.append(new_meta_annotation)

    # Sort annotations by scene and sample number
    annotations = sorted(annotations, key=lambda x: (
        int(x['rgb_path'].split('_')[-2]), 
        int(x['rgb_path'].split('_')[-1].split('.')[0])
    ))

    # Save annotations to JSON file
    with open(data_save_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    print(f"Generated {len(annotations)} annotations and saved to {data_save_path}")
    print(f"RGB images saved to: {rgb_output_dir}")
    # delete rgb_image
    if os.path.exists(rgb_temp_dir):
        shutil.rmtree(rgb_temp_dir)
if __name__ == "__main__":
    main()