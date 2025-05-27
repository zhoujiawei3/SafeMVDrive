from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import copy
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json
import os
import argparse
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
import hashlib
import shutil


def matrix_to_translation_rotation(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to translation vector and rotation quaternion
    
    Parameters:
        transformation_matrix: Transformation matrix with shape [4, 4]
    
    Returns:
        translation: Translation vector with shape [3,]
        rotation_quaternion: Quaternion object representing rotation
    """
    # Extract translation vector (position)
    translation = transformation_matrix[:3, 3]
    
    # Extract rotation matrix (3x3 part)
    rotation_matrix = transformation_matrix[:3, :3]
    
    # Convert rotation matrix to quaternion
    rotation_quaternion = Quaternion(matrix=rotation_matrix,rtol=1, atol=1)
    
    return translation, rotation_quaternion

def downsample_and_convert_transforms(matrices, source_fps=10, target_fps=2):
    """
    Downsample high frame rate transformation matrices and convert to position and rotation
    
    Parameters:
        matrices: numpy array with shape [num_frames, 4, 4], representing transformation matrix sequence
        source_fps: Original frame rate, default 10Hz
        target_fps: Target frame rate, default 2Hz
    
    Returns:
        translations: numpy array with shape [num_downsampled_frames, 3], representing position information
        rotations: numpy array with shape [num_downsampled_frames, 4], representing quaternion rotation information (w,x,y,z format)
    """
    # Calculate downsampling ratio
    downsample_ratio = source_fps / target_fps
    
    # Select indices of frames to keep
    indices = np.floor(np.arange(4, matrices.shape[0], downsample_ratio)).astype(int) # For trajectory prediction, keep the original first frame, so start from the 3rd frame as the next frame at 2hz
    
    # Ensure indices don't exceed array bounds
    indices = indices[indices < matrices.shape[0]]
    
    # Downsample matrices
    downsampled_matrices = matrices[indices]
    
    # Initialize result arrays
    num_frames = len(indices)
    translations = np.zeros((num_frames, 3))
    rotations = np.zeros((num_frames, 4))
    
    # Extract position and rotation from transformation matrices
    for i, matrix in enumerate(downsampled_matrices):
        # Use provided function to extract translation vector and rotation quaternion
        translation, quaternion = matrix_to_translation_rotation(matrix)
        
        # Store translation vector
        translations[i] = translation
        
        # Store quaternion (w,x,y,z format)
        rotations[i] = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
    
    return translations, rotations

def remove_before_first_slash(path):
    index = path.find('/')
    if index != -1:
        return path[index + 1:]
    return path

def upsample_and_convert_transforms(matrices, source_fps=10, target_fps=12,ego_first_translation=None, ego_first_roation=None):
    """
    Upsample low frame rate transformation matrices and convert to position and rotation, considering additional initial frame (0 second) information
    
    Parameters:
        matrices: numpy array with shape [num_frames, 4, 4], representing transformation matrix sequence (starting from 0.1 seconds)
        source_fps: Original frame rate, default 10Hz
        target_fps: Target frame rate, default 12Hz
        ego_first_translation: Translation vector with shape [3,], representing position at 0 seconds
        ego_first_rotation: Quaternion with shape [4,] (w,x,y,z format) or Quaternion object, representing rotation at 0 seconds
    
    Returns:
        translations: numpy array with shape [num_upsampled_frames, 3], representing position information
        rotations: numpy array with shape [num_upsampled_frames, 4], representing quaternion rotation information (w,x,y,z format)
    """
    import numpy as np
    from pyquaternion import Quaternion
    
    # Process ego_first_rotation, ensure it's a Quaternion object
    if ego_first_roation is not None:
        if isinstance(ego_first_roation, np.ndarray) or isinstance(ego_first_roation, list):
            ego_first_quaternion = Quaternion(ego_first_roation[0], ego_first_roation[1], 
                                            ego_first_roation[2], ego_first_roation[3])
        elif isinstance(ego_first_roation, Quaternion):
            ego_first_quaternion = ego_first_roation
        else:
            raise TypeError("ego_first_rotation must be numpy array, list or Quaternion object")
    
    # Original time series (starting from 0.1 seconds)
    original_frames = matrices.shape[0]
    # Time axis starting from 0.1 seconds [0.1, 0.2, ..., 0.1 + (n-1)/10]
    original_times = np.array([0.1 + i/source_fps for i in range(original_frames)])
    
    # Calculate total frames after upsampling
    # Total duration in seconds
    total_duration = original_times[-1]
    # Upsampled time series including 0 seconds
    upsampled_times = np.arange(0, total_duration+0.0001, 1/target_fps)
    upsampled_frames = len(upsampled_times)
    
    # Initialize result arrays
    translations = np.zeros((upsampled_frames, 3))
    rotations = np.zeros((upsampled_frames, 4))
    
    # Set data for 0 seconds
    if ego_first_translation is not None and ego_first_roation is not None:
        translations[0] = ego_first_translation
        rotations[0] = np.array([ego_first_quaternion.w, ego_first_quaternion.x, 
                                ego_first_quaternion.y, ego_first_quaternion.z])
    
    # Extract translation and rotation from all original frames
    all_translations = []
    all_quaternions = []
    for matrix in matrices:
        translation, quaternion = matrix_to_translation_rotation(matrix)
        all_translations.append(translation)
        all_quaternions.append(quaternion)
    
    all_translations = np.array(all_translations)
    
    # Add 0 seconds data to original sequence
    if ego_first_translation is not None and ego_first_roation is not None:
        ego_first_translation_np = np.array(ego_first_translation).reshape(1, 3)
        all_translations = np.vstack([ego_first_translation_np, all_translations])
        all_quaternions.insert(0, ego_first_quaternion)
        original_times = np.insert(original_times, 0, 0.0)
    
    # Perform interpolation for each upsampled time point
    for i, t in enumerate(upsampled_times):
        # Skip if it's 0 seconds and already set
        if t == 0 and ego_first_translation is not None:
            continue
            
        # Find the index of original frame interval containing t
        if t >= original_times[-1]:  # If it's the last time point
            translations[i] = all_translations[-1]
            rotations[i] = np.array([all_quaternions[-1].w, all_quaternions[-1].x, 
                                    all_quaternions[-1].y, all_quaternions[-1].z])
            continue
            
        idx = np.searchsorted(original_times, t, side='right') - 1
        next_idx = idx + 1
        
        # Ensure indices are within valid range
        if next_idx >= len(original_times):
            next_idx = len(original_times) - 1
            
        # Calculate interpolation ratio
        if idx == next_idx:  # Exact match with original frame
            amount = 0
        else:
            amount = (t - original_times[idx]) / (original_times[next_idx] - original_times[idx])
        
        # Linear interpolation for translation vector
        translations[i] = all_translations[idx] + amount * (all_translations[next_idx] - all_translations[idx])
        
        # Spherical linear interpolation for rotation quaternion
        interp_quaternion = Quaternion.slerp(
            q0=all_quaternions[idx],
            q1=all_quaternions[next_idx],
            amount=amount
        )
        rotations[i] = np.array([interp_quaternion.w, interp_quaternion.x, 
                                interp_quaternion.y, interp_quaternion.z])
    
    return translations, rotations

def copy_file_create_dir(source_path, destination_path):
    """
    Copy file from source path to destination path, create destination folder if it doesn't exist
    
    Parameters:
        source_path (str): Full path of source file
        destination_path (str): Full path of destination file
    
    Returns:
        bool: Whether operation was successful
    """
    try:
        # Get destination folder path
        destination_dir = os.path.dirname(destination_path)
        
        # Create destination folder if it doesn't exist (including all necessary parent folders)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print(f"Created destination folder: {destination_dir}")
        
        # Copy file
        shutil.copy2(source_path, destination_path)
        print(f"Successfully copied file from {source_path} to {destination_path}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Source file {source_path} does not exist")
        return False
    except PermissionError:
        print("Error: Insufficient permissions for operation")
        return False
    except Exception as e:
        print(f"Error occurred during copying: {e}")
        return False

def generate_token(key, data):
    """
    Generate an MD5 hash token by combining two strings
    
    Parameters:
        key: First input string
        data: Second input string
    
    Returns:
        Generated MD5 hash value (hexadecimal string)
    """
    # Create MD5 hash object
    if key == '':
        return ''
    obj = hashlib.md5(str(key).encode('utf-8'))
    
    # Update hash object, add second string
    obj.update(str(data).encode('utf-8'))
    
    # Get hexadecimal representation of hash value
    result = obj.hexdigest()
    
    return result

def get_first_frame_index(nusc,scene,sample_token,factor=5):
    first_frame_in_scene_token = scene['first_sample_token']
    for qwq in range(40):
        if first_frame_in_scene_token == sample_token:
            break
        sample = nusc.get('sample', first_frame_in_scene_token)
        first_frame_in_scene_token = sample['next']
    start_frame_index=qwq*factor+1
    return start_frame_index

def generate_sample_with_ann_and_scene_and_text_caption(args, nusc, trajectory_datas,text_caption_time_datas,text_caption_datas):
    new_sample_list = []
    new_sample_annotation_list = []
    new_scene_list= []
    new_text_caption_time_dict = {}
    new_text_caption_dict = {}
    sensor_list= ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    for trajectory_data in tqdm(trajectory_datas):
        # First update scene

        # Build a scene for each trajectory_data
        if 'collision_type' in trajectory_data:
            collision = trajectory_data["collision_type"]
        else:
            collision='test_crash'
        if "adv_id" in trajectory_data:
            adv_id = trajectory_data["adv_id"]
        else:
            adv_id = trajectory_data["adv_token"]
        first_sample_token= trajectory_data["sample_token"]
        # Confirm that there are 20 samples following in the nuscenes dataset
        sample_first= nusc.get("sample", first_sample_token)

        scene_token = sample_first["scene_token"]
        scene = nusc.get("scene", scene_token)

        start_frame_index=get_first_frame_index(nusc,scene,first_sample_token,factor=5)
        # Update scene number and scene name
        new_scene = copy.deepcopy(scene)
        new_scene["token"] = generate_token(new_scene["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        new_scene["name"] = new_scene["name"]+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}"
        new_scene["nbr_samples"] =19
        new_scene["first_sample_token"] = generate_token(sample_first["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        

        # Update text_caption_time and text_caption, original key is scene_camera = "{}|{}".format(scene, sensor["channel"])
        for sensor in sensor_list:
            scene_camera = "{}|{}".format(scene["token"], sensor)
            new_scene_camera = "{}|{}".format(new_scene["token"], sensor)
            if scene_camera in text_caption_time_datas:
                new_text_caption_time_dict[new_scene_camera] = text_caption_time_datas[scene_camera]
                for timestamp in text_caption_time_datas[scene_camera]:
                    scene_camera_timestamp = "{}|{}".format(scene_camera, timestamp)
                    new_scene_camera_timestamp = "{}|{}".format(new_scene_camera, timestamp)
                    new_text_caption_dict[new_scene_camera_timestamp] = text_caption_datas[scene_camera_timestamp]
        
        # Update sample (initial frame)
        key_frame_anns = [nusc.get('sample_annotation', token) for token in sample_first['anns']]
        instance_token_in_first_frame_anns=[ann["instance_token"] for ann in key_frame_anns]
        new_sample_first=copy.deepcopy(sample_first)
        new_sample_first["token"] = generate_token(new_sample_first["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        new_sample_first["prev"] = ''
        new_sample_first["next"] = generate_token(new_sample_first["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        new_sample_first["scene_token"] = new_scene["token"]
        new_sample_list.append(new_sample_first)

        # Update annotations for initial frame
        new_key_frame_anns = copy.deepcopy(key_frame_anns)
        first_timestamp=new_sample_first['timestamp']
        for ann in new_key_frame_anns:
            ann["token"] = generate_token(ann["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            ann["sample_token"] = new_sample_first["token"]
            ann["prev"] = ''
            ann["next"] = generate_token(ann["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            new_sample_annotation_list.append(ann)
        
        nearby_vehicle_annotation_dict={}
        for i in range(18):
            sample_token = nusc.get("sample", first_sample_token)["next"]
            assert sample_token !='', "Insufficient 10s"
            first_sample_token = sample_token
            sample = nusc.get("sample", sample_token)
            new_sample = copy.deepcopy(sample)
            new_sample["token"] = generate_token(new_sample["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            new_sample["prev"] = generate_token(new_sample["prev"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            if args.dataset_type!= 'oracle':
                new_sample['timestamp']=int(first_timestamp+ (i+1) / 2.0 * 1000000)
            if i == 17:
                new_sample["next"] = ''
            else:
                new_sample["next"] = generate_token(new_sample["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            
            new_sample["scene_token"] = new_scene["token"]
            new_sample_list.append(new_sample)            
            key_frame_anns = [nusc.get('sample_annotation', token) for token in new_sample['anns']]

            new_key_frame_anns = copy.deepcopy(key_frame_anns)
            for ann in new_key_frame_anns:
                ann["token"] = generate_token(ann["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                ann["sample_token"] = new_sample["token"]
                ann["prev"] = generate_token(ann["prev"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                if i == 17:
                    ann["next"] = ''
                else:
                    ann["next"] = generate_token(ann["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")

                # Check if it appears in the first frame, skip if not
                if args.dataset_type == 'adversarial_trajectory':
                    if ann["instance_token"] not in instance_token_in_first_frame_anns:
                        continue

                # Check if instance_token appears in vehicle_translation and vehicle_rotation, replace if found
                    for instance_annotations in trajectory_data["nearby_vehicle_translation"]:
                        instance_token=next(iter(instance_annotations))
                        vehicle_translation = instance_annotations[instance_token]
                        if instance_token == ann["instance_token"]:
                            ann["translation"] = vehicle_translation[i].tolist()
                            need_to_continue=False
                            if ann['next'] =='' and i!=17:
                                # Handle cases where there are no subsequent samples
                                ann["next"] = generate_token(instance_token, f"_frame_{i+1}_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                                nearby_vehicle_annotation_dict[instance_token] = ann
                            
                            break
                    
                    for instance_annotations in trajectory_data["nearby_vehicle_rotation"]:
                        instance_token=next(iter(instance_annotations))
                        vehicle_rotation = instance_annotations[instance_token]
                        if instance_token == ann["instance_token"]:
                            ann["rotation"] = vehicle_rotation[i].tolist()
                            break

                new_sample_annotation_list.append(ann)
            # For instance_tokens in trajectory_data that don't appear in annotations, generate new sample_annotations
            if args.dataset_type == 'adversarial_trajectory':
                for instance_annotations in trajectory_data["nearby_vehicle_translation"]:
                    instance_token=next(iter(instance_annotations))
                    vehicle_translation = instance_annotations[instance_token]

                    if instance_token not in [ann["instance_token"] for ann in new_key_frame_anns]:
                        if instance_token in nearby_vehicle_annotation_dict:
                            ann = copy.deepcopy(nearby_vehicle_annotation_dict[instance_token])
                            ann["prev"] = ann["token"]
                            ann["token"] = generate_token(instance_token, f"_frame_{i}_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                            if i == 17:
                                ann["next"] = ''
                            else:
                                ann["next"] = generate_token(instance_token, f"_frame_{i+1}_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                            ann["sample_token"] = new_sample["token"]
                            ann["translation"] = vehicle_translation[i].tolist()
                            
                            # Find rotation
                            for j in trajectory_data["nearby_vehicle_rotation"]:
                                instance_token_j=next(iter(j))
                                if instance_token_j == instance_token:
                                    vehicle_rotation = j[instance_token_j]
                                    ann["rotation"] = vehicle_rotation[i].tolist()
                                    break
                            ann["instance_token"] = instance_token
                            new_sample_annotation_list.append(ann)
        new_scene["last_sample_token"] = new_sample["token"]
        new_scene_list.append(new_scene)

    return new_sample_list, new_sample_annotation_list,new_scene_list, new_text_caption_time_dict, new_text_caption_dict




def generate_sample_data_with_pose(args, nusc, trajectory_datas):
    """
    Generate new sample data entries based on trajectory information
    
    Parameters:
        args: Command line arguments
        nusc: NuScenes dataset object
        trajectory_datas: List of trajectory data dictionaries
    
    Returns:
        new_sample_data_list: List of new sample data entries
        new_ego_pose_list: List of new ego pose entries
    """
    new_sample_data_list = []
    new_ego_pose_list = []
    sensor_modalities = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                            "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT","LIDAR_TOP"]
    
    for trajectory_data in tqdm(trajectory_datas):
        # Get collision type and adversarial token for generating unique IDs
        # sample_data_count=0
        if 'collision_type' in trajectory_data:
            collision = trajectory_data["collision_type"]
        else:
            collision='test_crash'
        if "adv_id" in trajectory_data:
            adv_id = trajectory_data["adv_id"]
        else:
            adv_id = trajectory_data["adv_token"]
        first_sample_token = trajectory_data["sample_token"]
        
        # Get the first sample and its related data
        sample_first = nusc.get("sample", first_sample_token)
        # next_sample_token = sample_first["next"]
        scene_token = sample_first["scene_token"]
        scene = nusc.get("scene", scene_token)
        start_frame_index=get_first_frame_index(nusc,scene,first_sample_token,factor=5)
        first_time_stamp=sample_first['timestamp']
        key_sample_token=first_sample_token
        for i in range(19):
            key_sample=nusc.get("sample",key_sample_token)
            for sensor in sensor_modalities:
                this_sample_data_token = key_sample["data"][sensor]
                this_sample_data = nusc.get('sample_data',this_sample_data_token)
                for frame_idx in range(6):
                    if frame_idx == 0:
                        frame0 = copy.deepcopy(this_sample_data)
                        if i==0:
                            frame0['prev']=''
                        else:
                            last_key_sample_token=key_sample['prev']
                            last_key_sample=nusc.get('sample',last_key_sample_token)
                            last_key_sample_data_token=last_key_sample["data"][sensor]
                            frame0['prev'] = generate_token(last_key_sample_data_token,f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")+'5'
                        frame0['next'] = generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + '1'
                        frame0['token'] = generate_token(this_sample_data["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        frame0["sample_token"] = generate_token(this_sample_data["sample_token"],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        frame0["timestamp"] = first_time_stamp+i/2.0*1000000+frame_idx/12.0 * 1000000
                        frame0['ego_pose_token'] = frame0['token']
                        if i==18:
                            frame0['next']=''
                        if sensor == 'LIDAR_TOP':
                            frame0['filename'] = "new_"+this_sample_data['filename'][:-8]+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}.pcd.bin"
                        else:
                            frame0['filename'] = "new_"+this_sample_data['filename'][:-4]+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}."+this_sample_data['filename'].split(".")[-1]
                        new_sample_data_list.append(frame0)

                        ego_pose={}
                        ego_pose['token']=frame0['ego_pose_token']
                        ego_pose['timestamp']=frame0['timestamp']
                        if args.dataset_type == 'adversarial_trajectory':
                            ego_pose["translation"] = trajectory_data["ego_translation"][i*6+frame_idx].tolist()
                            ego_pose["rotation"] = trajectory_data["ego_rotation"][i*6+frame_idx].tolist()
                        elif args.dataset_type == 'original_trajectory':
                            ego_pose['translation'] = nusc.get('ego_pose',nusc.get('sample_data', this_sample_data['token'])['ego_pose_token'])['translation']
                            ego_pose['rotation'] = nusc.get('ego_pose',nusc.get('sample_data', this_sample_data['token'])['ego_pose_token'])['rotation']
                        else:
                            print(args.dataset_type)
                            raise Exception("wrong type") 
                        new_ego_pose_list.append(ego_pose)

                        #os.copy
                        origin_file_path=os.path.join(args.data_path,this_sample_data['filename'])
                        destination_file_path=os.path.join(args.data_path,frame0['filename'])
                        copy_file_create_dir(origin_file_path,destination_file_path)
                        
                        if i==18:
                            break
                        
                    else:
                        extra_sample_data = copy.deepcopy(this_sample_data)
                        extra_sample_data['token']=generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + str(frame_idx)
                        extra_sample_data['sample_token'] = generate_token(key_sample['next'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        if frame_idx==1:
                            extra_sample_data['prev']=generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        else:
                            extra_sample_data['prev']=generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + str(frame_idx-1)
                        if frame_idx==5:
                            next_key_sample_token= key_sample['next']
                            next_key_sample=nusc.get('sample',next_key_sample_token)
                            next_key_sample_data_token=next_key_sample["data"][sensor]
                            extra_sample_data['next'] = generate_token(next_key_sample_data_token,f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        else:
                            extra_sample_data['next'] = generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + str(frame_idx+1)
                        extra_sample_data['timestamp']= first_time_stamp+i/2.0*1000000+frame_idx/12.0 * 1000000
                        
                        append_sample_data = copy.deepcopy(this_sample_data)
                        for a in range(frame_idx):
                            # try:
                            if nusc.get('sample_data', append_sample_data['next'])['is_key_frame']:
                                break
                            append_sample_data = nusc.get('sample_data', append_sample_data['next'])
                        extra_sample_data['ego_pose_token'] = extra_sample_data['token']
                        extra_sample_data['calibrated_sensor_token'] = nusc.get('sample_data', append_sample_data['token'])['calibrated_sensor_token']
                        if sensor == 'LIDAR_TOP':
                            extra_sample_data['filename'] = 'new_sweeps/'+remove_before_first_slash(this_sample_data['filename'][:-8])+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}_"+str(frame_idx)+'.pcd.bin'
                        else:
                            extra_sample_data['filename'] = 'new_sweeps/'+remove_before_first_slash(this_sample_data['filename'][:-4])+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}_"+str(frame_idx)+'.'+this_sample_data['filename'].split(".")[-1]
                        extra_sample_data['is_key_frame'] = False

                        new_sample_data_list.append(extra_sample_data)

                        ego_pose={}
                        ego_pose['token']=extra_sample_data['ego_pose_token']
                        ego_pose['timestamp']=extra_sample_data['timestamp']
                        if args.dataset_type == 'adversarial_trajectory':
                            ego_pose["translation"] = trajectory_data["ego_translation"][i*6+frame_idx].tolist()
                            ego_pose["rotation"] = trajectory_data["ego_rotation"][i*6+frame_idx].tolist()
                        elif args.dataset_type == 'original_trajectory':
                            ego_pose['translation'] = nusc.get('ego_pose',nusc.get('sample_data', append_sample_data['token'])['ego_pose_token'])['translation']
                            ego_pose['rotation'] = nusc.get('ego_pose',nusc.get('sample_data', append_sample_data['token'])['ego_pose_token'])['rotation']
                        else:
                            raise Exception("wrong type") 
                        
                        origin_file_path=os.path.join(args.data_path,nusc.get('sample_data', append_sample_data['token'])['filename'])
                        destination_file_path=os.path.join(args.data_path,extra_sample_data['filename'])
                        copy_file_create_dir(origin_file_path,destination_file_path)

                        new_ego_pose_list.append(ego_pose)
            key_sample_token=key_sample['next']
    return new_sample_data_list, new_ego_pose_list       
def save_json(args, new_sample_list, new_sample_annotation_list,new_scene_list, new_sample_data_list, new_ego_pose_list):
    out_dir = os.path.join(args.data_path, "v1.0-collision")
    os.makedirs(out_dir, exist_ok=True)
    
    print('Saving new samples, list length: {}'.format(len(new_sample_list)))
    with open(os.path.join(out_dir, 'sample.json'), 'w') as f:
        json.dump(new_sample_list, f, indent=4)

    print('Saving new sample annotation, list length: {}'.format(len(new_sample_annotation_list)))
    with open(os.path.join(out_dir, 'sample_annotation.json'), 'w') as f:
        json.dump(new_sample_annotation_list, f, indent=4)

    print('Saving new scene data, list length: {}'.format(len(new_scene_list)))
    with open(os.path.join(out_dir, 'scene.json'), 'w') as f:
        json.dump(new_scene_list, f, indent=4)

    print('Saving new sample data, list length: {}'.format(len(new_sample_data_list)))
    with open(os.path.join(out_dir, 'sample_data.json'), 'w') as f:
        json.dump(new_sample_data_list, f, indent=4)
    
    print('Saving new ego pose data, list length: {}'.format(len(new_ego_pose_list)))
    with open(os.path.join(out_dir, 'ego_pose.json'), 'w') as f:
        json.dump(new_ego_pose_list, f, indent=4)

    
    # Copy other required JSON files
    misc_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor', 'log', 'map']
    for misc_name in misc_names:
        p_misc_name = os.path.join(out_dir, misc_name + '.json')
        if not os.path.exists(p_misc_name):
            source_path = os.path.join(args.data_path, args.data_version, misc_name + '.json')
            os.system('cp {} {}'.format(source_path, p_misc_name))
    
    return out_dir
    
        
def parse_args():

    parser = argparse.ArgumentParser(description="NuScenes Trajectory Annotator")
    parser.add_argument("--data-path", type=str, default="../nuscenes", help="Path to nuScenes data")
    parser.add_argument("--data-version", type=str, default="v1.0-trainval", 
                        help="NuScenes dataset version")
    parser.add_argument("--trajectory-file", type=str, 
                        help="Input JSON file with trajectory information")
    parser.add_argument("--text_caption_time_json", type=str, default="text_description/nuscenes/nuscenes_v1.0-trainval_caption_v2_times_val.json")
    parser.add_argument("--text_caption_json", type=str, default="text_description/nuscenes/nuscenes_v1.0-trainval_caption_v2_val.json")
    parser.add_argument("--text_caption_output_dir", type=str, default="text_description/collide")
    parser.add_argument('--dataset_type', type=str, default="adversarial_trajectory", help='有三种一种oracle(只改变token),一种original_trajectory (改变token和timestamp),还有一种是adversarial_trajectory_and_sample(改变token和timestamp和trajectory)')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.data_version, dataroot=args.data_path, verbose=False)
    
    print(f"Loading trajectory data from {args.trajectory_file}...")
    with open(args.trajectory_file, 'r') as f:
        trajectory_datas = json.load(f)
    with open(args.text_caption_time_json, 'r') as f:
        text_caption_time_datas = json.load(f)
    with open(args.text_caption_json, 'r') as f:
        text_caption_datas = json.load(f)
    for trajectory_data in trajectory_datas:
        trajectory_data["nearby_vehicle_translation"] = []
        trajectory_data["nearby_vehicle_rotation"] = []
        predict_world_trajectory = trajectory_data['predict_world_trajectory']
        

        for data in predict_world_trajectory.items():   
            if data[0]=="ego":
                # Get ego translation and rotation
                sample_token = trajectory_data['sample_token']
                sample = nusc.get("sample", sample_token)
                sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                ego_first_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
                ego_first_translation = ego_first_pose["translation"]
                ego_first_rotation = ego_first_pose["rotation"]
                # Since ego_pose is 12Hz, we need to upsample. The output here includes translation and rotation for sample_token
                translations, rotations = upsample_and_convert_transforms(np.array(data[1]), source_fps=10, target_fps=12,ego_first_translation=ego_first_translation, ego_first_roation=ego_first_rotation)
                trajectory_data["ego_translation"] = translations
                trajectory_data["ego_rotation"] = rotations
            else:
                # Since annotation is 2Hz, we need to downsample. The output here does not include translation and rotation for sample_token
                translations, rotations = downsample_and_convert_transforms(np.array(data[1]), source_fps=10, target_fps=2) 
                trajectory_data["nearby_vehicle_translation"].append({data[0]: translations})
                trajectory_data["nearby_vehicle_rotation"].append({data[0]: rotations})
    

    # Process trajectory data and create new version
    new_sample_list, new_sample_annotation_list,new_scene_list,new_text_caption_time_dict,new_text_caption_dict = generate_sample_with_ann_and_scene_and_text_caption(args, nusc, trajectory_datas,text_caption_time_datas,text_caption_datas)
    new_sample_data_list,new_ego_pose_list = generate_sample_data_with_pose(args, nusc, trajectory_datas)
    out_dir = save_json(args, new_sample_list, new_sample_annotation_list,new_scene_list, new_sample_data_list, new_ego_pose_list)


    # Save text caption data
    print('Saving text caption data...')
    if not os.path.exists(args.text_caption_output_dir):
        os.makedirs(args.text_caption_output_dir, exist_ok=True)
    with open(os.path.join(args.text_caption_output_dir, "nuscenes_v1.0-trainval_caption_v2_val_collision.json"), 'w') as f:
        json.dump(new_text_caption_dict, f, indent=4)
    with open(os.path.join(args.text_caption_output_dir, "nuscenes_v1.0-trainval_caption_v2_times_val_collision.json"), 'w') as f:
        json.dump(new_text_caption_time_dict, f, indent=4)
    # Generate sample_data list
    print('processing sample data lists...')
    # final_path = os.path.join(args.data_path, "collide")
    # os.system('cp -r {} {}'.format(out_dir, final_path))
    print(f"New nuScenes version with trajectory annotations created at: {out_dir}")
    
    