"""A script for evaluating closed-loop simulation"""
import argparse
from symbol import star_expr
import numpy as np
import json
import random
import yaml
import importlib
from collections import Counter
from pprint import pprint

import os
import torch
from nuscenes import NuScenes
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvUnifiedBuilder, EnvL5Builder
from tbsim.utils.guidance_loss import verify_guidance_config_list
from tqdm import tqdm
from tbsim.policies.wrappers import (
    RolloutWrapper,
    Pos2YawWrapper,
)
import imageio
import os
from tbsim.utils.tensor_utils import map_ndarray
import tbsim.utils.tensor_utils as TensorUtils
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import re
def save_images_to_video(image_dir, video_path, fps=2):
    """

    :param image_dir: 
    :param video_path:
    :param fps: 
    """
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    images = []
    for i in range(20):
        image_path = os.path.join(image_dir, '%03d'%i+".png")
        images.append(imageio.imread(image_path))
    
    with imageio.get_writer(os.path.join(video_path, 'video.mp4'), fps=fps) as writer:
        for image in images:
            writer.append_data(image)


dataroot = '../nuscenes'

output_dataset=dataroot.split('/')[-1]
# nuscenes = NuScenes('v1.0-mini', dataroot=dataroot)
nuscenes = NuScenes('v1.0-trainval', dataroot=dataroot)
def get_matrix_from_3x3_to_4x4(world_from_agent,global_from_ego):

    time_steps = world_from_agent.shape[0]  
    ego_world_from_agent_4x4 = np.zeros((time_steps, 4, 4))

    for t in range(time_steps):

        ego_world_from_agent_4x4[t, 0:2, 0:2] = world_from_agent[t, 0:2, 0:2]
        

        ego_world_from_agent_4x4[t, 0:2, 3] = world_from_agent[t, 0:2, 2]
        

        ego_world_from_agent_4x4[t, 2, 2] = global_from_ego[2, 2]  
        ego_world_from_agent_4x4[t, 2, 3] = global_from_ego[2, 3]  
        

        ego_world_from_agent_4x4[t, 0, 2] = global_from_ego[0, 2]
        ego_world_from_agent_4x4[t, 1, 2] = global_from_ego[1, 2]
        ego_world_from_agent_4x4[t, 2, 0] = global_from_ego[2, 0]
        ego_world_from_agent_4x4[t, 2, 1] = global_from_ego[2, 1]
        

        ego_world_from_agent_4x4[t, 3, :] = global_from_ego[3, :]
    return ego_world_from_agent_4x4
def get_predict_world_trajectory(scenes,info):
    predict_trajectory={}
    sample_token=scenes[0]["sample_token"]
    
    sample= nuscenes.get('sample', sample_token)
    lidar_sample_data_token= sample['data']["LIDAR_TOP"]
    lidar_sample_data= nuscenes.get('sample_data', lidar_sample_data_token)
    ego_pose= nuscenes.get('ego_pose', lidar_sample_data['ego_pose_token'])
    global_from_ego=transform_matrix(ego_pose['translation'], Quaternion(ego_pose["rotation"]), inverse=False) #[4,4]

    info_buffer =info["buffer"]
    world_from_agent = info_buffer[0]["world_from_agent"]
    ego_world_from_agent = world_from_agent[0] #[100,3,3]

    ego_world_from_agent_4x4=get_matrix_from_3x3_to_4x4(ego_world_from_agent,global_from_ego) #[100,4,4]
    
    predict_trajectory["ego"] = ego_world_from_agent_4x4.tolist()
    agent_name = info_buffer[0]["agent_name"]

    annotation= sample['anns']

    for i in range(len(agent_name)):
        
        instance_token=agent_name[i][0][0]
        if instance_token == "ego":
            pass
        for j in range(len(annotation)):
            anno= nuscenes.get('sample_annotation', annotation[j])
            if anno['instance_token'] == instance_token:
                translation=anno['translation']
                rotation=anno['rotation']
                agent_global_from_ego=transform_matrix(translation, Quaternion(rotation), inverse=False)
                agent_world_from_agent=world_from_agent[i]#[100,3,3]
                agent_world_from_agent_4x4=get_matrix_from_3x3_to_4x4(agent_world_from_agent,agent_global_from_ego) #[100,4,4]
                predict_trajectory[instance_token] = agent_world_from_agent_4x4.tolist()
                break
    return predict_trajectory
def run_scene_editor(device,eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg,simulation_json,simulation_new_new_json,single_scene_index):
    assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"
        
    set_global_batch_type("trajdata")
    if eval_cfg.env == "nusc":
        set_global_trajdata_batch_env("nusc_trainval")
    elif eval_cfg.env == "trajdata":
        # assumes all used trajdata datasets use share same map layers
        set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0]) #eval_cfg.trajdata_source_test[0]=='nusc_train-val'

    # print(eval_cfg)

    # for reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)

    # device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    # create policy and rollout wrapper  
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    
    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer) 
    
    # print(exp_config.algo)
    # ----------------------------------------------------------------------------------
    policy_model = None
    print('policy', policy)
    if hasattr(policy, 'model'):
        policy_model = policy.model
    # Set evaluation time sampling/optimization parameters
    if eval_cfg.apply_guidance:
        if eval_cfg.eval_class in ['SceneDiffuser', 'Diffuser', 'TrafficSim', 'BC', 'HierarchicalSampleNew']:
            policy_model.set_guidance_optimization_params(eval_cfg.guidance_optimization_params)
        if eval_cfg.eval_class in ['SceneDiffuser', 'Diffuser']:
            policy_model.set_diffusion_specific_params(eval_cfg.diffusion_specific_params)
    # ----------------------------------------------------------------------------------

    # create env
    if eval_cfg.env == "nusc": 
        env_builder = EnvNuscBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        if "parse_obs" in exp_config.env.data_generation_params:
            parse_obs=exp_config.env.data_generation_params.parse_obs
        else:
            parse_obs=True
        env = env_builder.get_env(parse_obs=parse_obs)
    elif eval_cfg.env == "trajdata":
        
        env_builder = EnvUnifiedBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        env = env_builder.get_env()

    else:
        raise NotImplementedError("{} is not a valid env".format(eval_cfg.env))

    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]

    heuristic_config = None
    use_ui = False
    if "ui" in eval_cfg.edits.editing_source:
        # TODO if using UI, initialize UI
        print("Using ONLY user interface to get scene edits...")
        use_ui = True
        raise NotImplementedError('UI')
    elif "heuristic" in eval_cfg.edits.editing_source:
        # verify heuristic args are valid
        if eval_cfg.edits.heuristic_config is not None:
            heuristic_config = eval_cfg.edits.heuristic_config
        else:
            heuristic_config = []

    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        # initialize rasterizer once for all scenes
        render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_test,
                                                  eval_cfg.trajdata_data_dirs,
                                                  future_sec=eval_cfg.future_sec,
                                                  history_sec=eval_cfg.history_sec,
                                                  raster_size=render_cfg['size'],
                                                  px_per_m=render_cfg['px_per_m'],
                                                  rebuild_maps=False,
                                                  cache_location='~/.unified_data_cache')

    result_stats = None

    # eval_scenes = eval_cfg.eval_scenes
    # eval_sample_token="9d77cf89d9114ec5a81a08f3a50e943b"
    # scene = nuscenes.get('scene', nuscenes.get('sample', eval_sample_token)['scene_token'])
    # scene_name = scene['name']
    # first_frame_in_scene_token = scene['first_sample_token']

    # for qwq in range(40):
    #     if first_frame_in_scene_token == eval_sample_token:
    #         break
    #     sample = nuscenes.get('sample', first_frame_in_scene_token)
    #     first_frame_in_scene_token = sample['next']
    # start_frame_index=qwq*5+1
    scene_i = 0
    total_scenes = len(simulation_new_new_json)
    # if simulation_new_new_json[0]['ref_agent'] is None:
    #     not_VLM_reason=True
    if eval_cfg.trajectory_output:
        trajectory_json = []
    with tqdm(total=total_scenes, desc="Processing scenes") as pbar:
        while scene_i < len(simulation_new_new_json):
            if eval_cfg.trajectory_output:
                trajectory_data={
                    "sample_token":None,
                    "collision":None,
                    "adv_token":None,
                    "predict_world_trajectory":None,
                    "action_positions":None,
                    "action_rotations":None
                }
            if scene_i + eval_cfg.num_scenes_per_batch > len(simulation_new_new_json) or scene_i >= single_scene_index:
                eval_cfg.unlock()
                eval_cfg.num_scenes_per_batch = 1
                eval_cfg.lock()
                scenes= simulation_new_new_json[scene_i:scene_i + eval_cfg.num_scenes_per_batch]
                scene_i += eval_cfg.num_scenes_per_batch
            else:
                scenes = simulation_new_new_json[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
                scene_i += eval_cfg.num_scenes_per_batch
            

            # scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
            # scene_name=scene['scene_name']
            # start_frame_index=scene['start_frame_index']


            
            start_frame_index = [scene['start_frame_index'] for scene in scenes]
            scene_indices = [env.dataset.get_index_from_scene_name(scene['scene_name']) for scene in scenes]
            # print("rgb_path", [scene['rgb_path'] for scene in scenes])
            print('scene_indices', scene_indices)
            print('start_frame_index', start_frame_index)
            # check to make sure all the scenes are valid at starting step
            scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=start_frame_index)
            scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            if len(scene_indices) == 0:
                print('no valid scenes in this batch, skipping...')
                torch.cuda.empty_cache()
                continue


            # if requested, split each scene up into multiple simulations
            # start_frame_index = [start_frame_index]
            if eval_cfg.num_sim_per_scene > 1:
                start_frame_index = []
                for si in range(len(scene_indices)):
                    cur_scene = env._current_scenes[si].scene
                    sframe = exp_config.algo.history_num_frames+1
                    # want to make sure there's GT for the full rollout
                    eframe = cur_scene.length_timesteps - eval_cfg.num_simulation_steps
                    scene_frame_inds = np.linspace(sframe, eframe, num=eval_cfg.num_sim_per_scene, dtype=int).tolist()
                    start_frame_index.append(scene_frame_inds)

            # how many sims to run for the current batch of scenes
            start_frame_index = [[index] for index in start_frame_index]
            print('Starting frames in current scenes:', start_frame_index)
            for ei in range(eval_cfg.num_sim_per_scene):
                guidance_config = None   # for the current batch of scenes
                constraint_config = None # for the current batch of scenes
                
                cur_start_frames = [scene_start[ei] for scene_start in start_frame_index]
                # double check all scenes are valid at the current start step
                scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=cur_start_frames)
                sim_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
                sim_start_frames = [sframe for sframe, sval in zip(cur_start_frames, scenes_valid) if sval]
                if len(sim_scene_indices) == 0:
                    torch.cuda.empty_cache()
                    continue

                if not use_ui:
                    # getting edits from either the config file or on-the-fly heuristics
                    if "config" in eval_cfg.edits.editing_source:
                        guidance_config = eval_cfg.edits.guidance_config
                        constraint_config  = eval_cfg.edits.constraint_config
                    if "heuristic" in eval_cfg.edits.editing_source:
                        # reset so that we can get an example batch to initialize guidance more efficiently
                        env.reset(scene_indices=scene_indices, start_frame_index=sim_start_frames)
                        ex_obs = env.get_observation()
                        # 
                        ego_index=np.where(np.array(ex_obs['agents']['agent_name']) == "ego")[0]
                        ego_curr_speed=ex_obs['agents']['curr_speed'][ego_index]
                        for count_speed,scene in enumerate(scenes):
                            scene['ego_init_speed']=ego_curr_speed[count_speed]
                        if obs_to_torch:
                            device = policy.device if device is None else device
                            ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)

                        # build heuristic guidance configs for these scenes
                        heuristic_guidance_cfg = compute_heuristic_guidance(heuristic_config,
                                                                            env,
                                                                            sim_scene_indices,
                                                                            sim_start_frames,
                                                                            example_batch=ex_obs['agents'],
                                                                            scenes=scenes,
                                                                            nusc=nuscenes,
                                                                            )
                                                                            
                        if len(heuristic_config) > 0:
                            # we asked to apply some guidance, but if heuristic determined there was no valid
                            #       guidance to apply (e.g. no social groups), we should skip these scenes.
                            valid_scene_inds = []
                            for sci, sc_cfg in enumerate(heuristic_guidance_cfg):
                                if len(sc_cfg) > 0:
                                    valid_scene_inds.append(sci)

                            # collect only valid scenes under the given heuristic config
                            heuristic_guidance_cfg = [heuristic_guidance_cfg[vi] for vi in valid_scene_inds]
                            sim_scene_indices = [sim_scene_indices[vi] for vi in valid_scene_inds]
                            sim_start_frames = [sim_start_frames[vi] for vi in valid_scene_inds]
                            # skip if no valid...
                            if len(sim_scene_indices) == 0:
                                print('No scenes with valid heuristic configs in this sim, skipping...')
                                torch.cuda.empty_cache()
                                continue

                        # add to the current guidance config



                        guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)
                else:
                    # TODO get guidance from the UI
                    # TODO for UI, get edits from user. loop continuously until the user presses
                    #       "play" or something like that then we roll out.
                    raise NotImplementedError()
            if len(sim_scene_indices) == 0:
                print('No scenes with valid heuristic configs in this scene, skipping...')
                torch.cuda.empty_cache()
                continue

            # remove agents from agent_collision guidance if they are in chosen gptcollision pair
            skip=True
            for si in range(len(guidance_config)):
                
                for i, cur_heur in enumerate(guidance_config[si]):
                    if cur_heur['name'] == 'collisiondiskLoss' or cur_heur['name'] == 'gptcollision':
                        agent_collision_heur_ind = i
                        skip=False
            if skip:
                for scene in scenes:
                    for k in simulation_json:
                        if k['sample_token']==scene['sample_token']:
                            k['ego_init_speed']=float(scene['ego_init_speed'])
                            k['skip']=True
                            torch.cuda.empty_cache()
                            pbar.update(eval_cfg.num_scenes_per_batch)
                            break
                continue
            for si in range(len(guidance_config)):
                if len(guidance_config[si]) > 0:
                    agent_collision_heur_ind = None
                    gpt_collision_heur_ind = None
                    for i, cur_heur in enumerate(guidance_config[si]):
                        if cur_heur['name'] == 'agent_collision':
                            agent_collision_heur_ind = i
                        elif cur_heur['name'] == 'gptcollision' or cur_heur['name'] == 'collisiondiskLoss':
                            gpt_collision_heur_ind = i
                    if agent_collision_heur_ind is not None and gpt_collision_heur_ind is not None:
                        ind1 = guidance_config[si][gpt_collision_heur_ind]['params']['target_ind']
                        ind2 = guidance_config[si][gpt_collision_heur_ind]['params']['ref_ind']
                        excluded_agents = [ind1, ind2]
                        guidance_config[si][agent_collision_heur_ind]['params']['excluded_agents'] = excluded_agents
                        print('excluded_agents', excluded_agents)

            # ----------------------------------------------------------------------------------
            # Sampling Wrapper leveraging most existing policy composer sampling interfaces
            from tbsim.policies.wrappers import NewSamplingPolicyWrapper
            if eval_cfg.eval_class in ['TrafficSim', 'HierarchicalSampleNew']:
                if scene_i == eval_cfg.num_scenes_per_batch or not isinstance(policy, NewSamplingPolicyWrapper):
                    policy = NewSamplingPolicyWrapper(policy, guidance_config)
                else:
                    policy.update_guidance_config(guidance_config)
            # ----------------------------------------------------------------------------------

            if eval_cfg.policy.pos_to_yaw:
                policy = Pos2YawWrapper(
                    policy,
                    dt=exp_config.algo.step_time,
                    yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
                )
            
            # right now assume control of full scene
            rollout_policy = RolloutWrapper(agents_policy=policy)


            stats, info, renderings = guided_rollout(
                env,
                rollout_policy,
                policy_model,
                n_step_action=eval_cfg.n_step_action,
                guidance_config=guidance_config,
                constraint_config=constraint_config,
                render=False, # render after the fact
                scene_indices=scene_indices,
                obs_to_torch=obs_to_torch,
                horizon=eval_cfg.num_simulation_steps,
                start_frames=sim_start_frames,
                eval_class=eval_cfg.eval_class,
                apply_guidance=eval_cfg.apply_guidance,
                scenes=scenes
            )    

            print(info["scene_index"])
            print(sim_start_frames)
            print(stats)
            if eval_cfg.trajectory_output:
                if scenes[0]["right"]==2 or scenes[0]["right"]==1 or eval_cfg.allTrajectory:
                    predict_world_trajectory=get_predict_world_trajectory(scenes,info)
                    action_positions= info["buffer"][0]["action_positions"].tolist()
                    action_yaws= info["buffer"][0]["action_yaws"].tolist()

                    trajectory_data["sample_token"]=scenes[0]["sample_token"]
                    trajectory_data["collision"]=scenes[0]["collision"]
                    trajectory_data["adv_token"]=scenes[0]["ref_agent_token"]
                    trajectory_data["predict_world_trajectory"]=predict_world_trajectory
                    trajectory_data["action_positions"]=action_positions
                    trajectory_data["action_rotations"]=action_yaws
                    trajectory_json.append(trajectory_data)
            # aggregate stats from the same class of guidance within each scene
            #       this helps parse_scene_edit_results
            guide_agg_dict = {}
            pop_list = []
            for k,v in stats.items():
                if k.split('_')[0] == 'guide':
                    guide_name = '_'.join(k.split('_')[:-1])
                    guide_scene_tag = k.split('_')[-1][:2]
                    canon_name = guide_name + '_%sg0' % (guide_scene_tag)
                    if canon_name not in guide_agg_dict:
                        guide_agg_dict[canon_name] = []
                    guide_agg_dict[canon_name].append(v)
                    # remove from stats
                    pop_list.append(k)
            for k in pop_list:
                stats.pop(k, None)
            # average over all of the same guide stats in each scene
            for k,v in guide_agg_dict.items():
                scene_stats = np.stack(v, axis=0) # guide_per_scenes x num_scenes (all are nan except 1)
                stats[k] = np.mean(scene_stats, axis=0)

            # aggregate metrics stats
            if result_stats is None:
                result_stats = stats
                result_stats["scene_index"] = np.array(info["scene_index"])
            else:
                for k in stats:
                    if k not in result_stats:
                        result_stats[k] = stats[k]
                    else:
                        result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
                result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

            # write stats to disk
            with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
                stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
                json.dump(stats_to_write, fp)

            if render_to_video or render_to_img:
                # high quality
                from tbsim.utils.scene_edit_utils import visualize_guided_rollout
                scene_cnt = 0
                scene_index_buffer=None
                for si, scene_buffer in zip(info["scene_index"], info["buffer"]):
                    # if scene_index_buffer is None or scene_index_buffer != si:
                    #     scene_index_buffer = si
                    #     scene_cnt = 0
                    viz_dir = os.path.join(eval_cfg.results_dir, "viz/")
                    invalid_guidance = guidance_config is None or len(guidance_config) == 0
                    invalid_constraint = constraint_config is None or len(constraint_config) == 0
                    visualize_guided_rollout(viz_dir, render_rasterizer, si, scene_buffer,
                                                guidance_config=None if invalid_guidance else guidance_config[scene_cnt],
                                                constraint_config=None if invalid_constraint else constraint_config[scene_cnt],
                                                fps=(1.0 / exp_config.algo.step_time),
                                                n_step_action=eval_cfg.n_step_action,
                                                viz_diffusion_steps=False,
                                                first_frame_only=render_to_img,
                                                sim_num=sim_start_frames[scene_cnt],
                                                save_every_n_frames=render_cfg['save_every_n_frames'],
                                                draw_mode=render_cfg['draw_mode'],generateType=scenes[scene_cnt]['generate_collision_type'])
                    if render_to_video:
                        hastarget=False
                        for guidance in guidance_config[scene_cnt]:
                            if guidance['name']=='collisiondiskLoss' or guidance['name']=='gptcollision':
                                hastarget=True
                                target=guidance['params']['ref_ind']
                                generateType=scenes[scene_cnt]['generate_collision_type']
                                if 'collision_type' in guidance['params']:
                                    collision_type=guidance['params']['collision_type']
                                else:
                                    collision_type="test_crash"
                        if hastarget:
                            save_images_to_video(viz_dir+'/'+si+'_'+'%04d'%sim_start_frames[scene_cnt]+'_adv%d'%(target)+'_desired_type'+collision_type+'_generateType_'+generateType, os.path.join(eval_cfg.results_dir, "video",si+'_'+'%04d'%sim_start_frames[scene_cnt]+'_adv%d'%(target)+'_desired_type'+collision_type+'_generateType_'+generateType), fps=2)
                        else:
                            save_images_to_video(viz_dir+'/'+si+'_'+'%04d'%sim_start_frames[scene_cnt], os.path.join(eval_cfg.results_dir, "video",si+'_'+'%04d'%sim_start_frames[scene_cnt]), fps=2)
                    scene_cnt += 1

            if data_to_disk and "buffer" in info:
                dump_episode_buffer(
                    info["buffer"],
                    info["scene_index"],
                    
                    sim_start_frames,
                    h5_path=eval_cfg.experience_hdf5_path
                )
            
            for scene in scenes:
                for k in simulation_json:
                    # if k['sample_token']==scene['sample_token'] and k['collision']==scene['collision']:
                    #     k['ego_init_speed']=float(scene['ego_init_speed'])
                    #     if scene['collision']==scene['generate_collision_type']:
                    #         k['reward'][scene['ref_agent']]=1
                    #     break

                    #如果是guidance loss不利用collisiontype
                    if k['sample_token']==scene['sample_token']:
                        k['ego_init_speed']=float(scene['ego_init_speed'])
                        if (scene['right']==2) or (scene['right']==1):
                            k['collision_dict'][scene['ref_agent']]=1
                        if k['collision']!='inference':
                            if k['collision']==scene['generate_collision_type']:
                                k['simulation_result'][scene['ref_agent_token']]=1
                        else:
                            k['simulation_result']=scene['generate_collision_type']

                    
                    
            # torch.cuda.empty_cache()
            pbar.update(eval_cfg.num_scenes_per_batch)
    with open(eval_cfg.simulation_file_path, "w") as fp:
        json.dump(simulation_json, fp,indent=4)
    if eval_cfg.trajectory_output:
        with open(eval_cfg.simulation_json_file_path, "w") as fp:
            json.dump(trajectory_json, fp,indent=4)

    collision_counter = 0
    for simulation in simulation_json:
        for k in simulation['collision_dict'].keys():
            if simulation['collision_dict'][k]==1:
                collision_counter+=1
    if simulation_json[0]['collision'] != 'inference':
        print('collision_counter', collision_counter/5)
    else:
        print('collision_counter', collision_counter)

    skip_counter=0
    for simulation in simulation_json:
        if simulation['skip']==True:
            skip_counter+=1
    print('skip_counter', skip_counter)
    collide_type_counter = {
            'A vehicle cuts in and collides with the ego vehicle': 0,
            'A vehicle rear-ends the ego vehicle': 0,
            'Ego vehicle rear-ends another vehicle': 0,
            'A vehicle has a head-on collision with the ego vehicle': 0,
            'A vehicle has a T-bone collision with the ego vehicle': 0
            }
    for simulation in simulation_json:
        if simulation['simulation_result']!={}:
            if simulation['collision'] !='inference':
                collide_type_counter[simulation['collision']]+=1
            else:
                if simulation['simulation_result'] != 'No':
                    collide_type_counter[simulation['simulation_result']]+=1
    print('collide_type_counter', collide_type_counter)
                
        
    
    

def dump_episode_buffer(buffer, scene_index, start_frames, h5_path):
    import h5py
    h5_file = h5py.File(h5_path, "a")

    for ei, si, scene_buffer in zip(start_frames, scene_index, buffer):
        for mk in scene_buffer:
            h5key = "/{}_{}/{}".format(si, ei, mk)
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--simulation_json_path",
        type=str,
        default=None,
        help="A json file containing all simulation scene"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "trajdata"],
        help="Which env to run editing in",
        required=True
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ------ for BITS ------
    parser.add_argument(
        "--planner_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--planner_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--predictor_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--predictor_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ----------------------

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--registered_name",
        type=str,
        default='trajdata_nusc_diff',
    )

    parser.add_argument(
        "--render_img",
        action="store_true",
        default=False,
        help="whether to only render the first frame of rollout"
    )

    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=2.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=5,
        help="saving videos while skipping every n frames"
    )

    parser.add_argument(
        "--draw_mode",
        type=str,
        default='action',
        help="['action', 'entire_traj', 'map']"
    )
    parser.add_argument(
        "--trajdata_source_test",
        type=str,
        default='val',
    )
    #
    # Editing options
    #
    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "ui", "none"],
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. UI will use interactive interface. \
              config and heuristic may be used together. If none, does not use edits."
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Which cuda device to use"
    )

    parser.add_argument(
        "--refix",
        type=str,
        default="test_collide",
        help="Which cuda device to use"
    )
    parser.add_argument(
        "--trajectory_output",
        action="store_true",
        default=True,
        help="whether to generate output json file"
    )
    parser.add_argument(
        "--second_simulation_path",
        type=str,
        default=None,
        help="A json file containing all simulation scene"
    )
    
    dataroot = '../nuscenes'

    output_dataset=dataroot.split('/')[-1]
    # nuscenes = NuScenes('v1.0-mini', dataroot=dataroot)
    nuscenes = NuScenes('v1.0-trainval', dataroot=dataroot)

    args = parser.parse_args()

    args = parser.parse_args()

    cfg = SceneEditingConfig(registered_name=args.registered_name)
    
    simulation_json = json.load(open(args.simulation_json_path, "r"))
    # simulation_json=simulation_json[:2]
    dir_name = os.path.dirname(args.simulation_json_path)
    base_name, ext = os.path.splitext(os.path.basename(args.simulation_json_path))

    new_file_name = f"{base_name}_{args.refix}{ext}"
    new_path = os.path.join(dir_name, new_file_name)
    cfg.simulation_file_path=new_path
    cfg.allTrajectory=args.second_simulation_path!=None
    if args.trajectory_output:
        new_json_file_name= f"{base_name}_{args.refix}-trajectory{ext}"
        cfg.simulation_json_file_path=os.path.join(dir_name, new_json_file_name)
    device = torch.device("cuda:"+str(args.cuda) if torch.cuda.is_available() else "cpu")

    if args.second_simulation_path is not None:
        second_simulation_json=json.load(open(args.second_simulation_path, "r"))
    print("original_simulation_length", len(simulation_json))
    for k in simulation_json:
        sample_token = k["sample_token"]
        scene = nuscenes.get('scene', nuscenes.get('sample', sample_token)['scene_token'])
        scene_name = scene['name']
        first_frame_in_scene_token = scene['first_sample_token']

        for qwq in range(40):
            if first_frame_in_scene_token == sample_token:
                break
            sample = nuscenes.get('sample', first_frame_in_scene_token)
            first_frame_in_scene_token = sample['next']
        start_frame_index=qwq*5+1
        k["scene_name"] = scene_name
        k["start_frame_index"] = start_frame_index
        k['simulation_result']={}
        k['collision_dict']={}
        
        if 'answer' in k:
            k['collision']='inference'
            match = re.search(r'\d+', k['answer'])
            
            if k['answer']!="" and match:
                answer_number=str(match.group())
                k['collision_dict'][answer_number]=0
        else:
            for agent in k['reward'].keys():
                k['collision_dict'][agent]=0
        k['skip']=False
    for k in simulation_json:
        simulation_name_same=0
        for j in simulation_json:
            if k['scene_name']==j['scene_name'] and k['start_frame_index']==j['start_frame_index']:
                simulation_name_same+=1
        if simulation_name_same>5:
            print("simulation_name_same", simulation_name_same)
            print("scene_name", k['scene_name'])
            print("start_frame_index", k['start_frame_index'])
            print("sample_token", k['sample_token'])



    simulation_new_json = []
    print("add_simulation_length", len(simulation_json))
    for k in simulation_json:
        
        meta_annotation=k.copy()
        next_json=False
        if 'answer' in k:
            match = re.search(r'\d+', k['answer'])
            if k['answer']!="" and match:
                answer_number=str(match.group())
                if 'ground_truth' in k: 
                    if k['ground_truth']!='':
                        if answer_number not in k['ground_truth']:
                            continue
                if answer_number not in k['token']:
                    continue
                meta_annotation["ref_agent"]=answer_number
                meta_annotation["ref_agent_reward"]=1
                meta_annotation["ref_agent_token"]=k['token'][answer_number]
                print("ref_agent_token", k['token'][answer_number])
                meta_annotation['right']=0
                meta_annotation['generate_collision_type']="No"
                simulation_new_json.append(meta_annotation.copy())
            continue
        else:
            if args.second_simulation_path is not None:
                second_sample_token_list=[i['sample_token'] for i in second_simulation_json]
                if k['sample_token'] not in second_sample_token_list:
                    continue
            
            for agent in k['reward'].keys():
                if k['reward'][agent]==1:
                    #for VLM reason
                    meta_annotation["ref_agent"]=agent
                    meta_annotation["ref_agent_reward"]=k['reward'][agent]
                    meta_annotation["ref_agent_token"]=k['token'][agent]
                    meta_annotation['right']=0
                    meta_annotation['generate_collision_type']="No"
                    simulation_new_json.append(meta_annotation.copy())
                    next_json=True
                    break
        if next_json:
            continue
        else:
            #for test crash
            meta_annotation["ref_agent"]=None
            meta_annotation["ref_agent_reward"]=None
            meta_annotation["ref_agent_token"]=None
            meta_annotation['right']=0
            meta_annotation['generate_collision_type']="No"
            has_same_scene_and_car=False
            for item in simulation_new_json:
                if item['scene_name']==meta_annotation['scene_name'] and item['ref_agent_token']==meta_annotation['ref_agent_token'] and item['start_frame_index']==meta_annotation['start_frame_index']:
                    has_same_scene_and_car=True
                    break
            if not has_same_scene_and_car:
                simulation_new_json.append(meta_annotation.copy())
    print("strange_simulation_length", len(simulation_new_json))
                


    from collections import defaultdict
    def sort_and_batch(simulation_json, numble_sample_per_batch):
        dict_json = {}
        for json in simulation_json:
            if json['scene_name'] not in dict_json:
                dict_json[json['scene_name']] = [json]
            else:
                dict_json[json['scene_name']].append(json)
        
        sorted_dict_json = [[k, v] for k, v in sorted(dict_json.items(), key=lambda x: len(x[1]), reverse=True)]
        output_json = []
        single_scene_index=0
        while len(sorted_dict_json) >= numble_sample_per_batch:
            for i in range(numble_sample_per_batch):
                output_json.append(sorted_dict_json[i][1].pop(0))
            single_scene_index+=numble_sample_per_batch
            sorted_dict_json = [[k, v] for k, v in sorted(sorted_dict_json, key=lambda x: len(x[1]), reverse=True)]
            sorted_dict_json = [[k, v] for k, v in sorted_dict_json if len(v) > 0]
        for k, v in sorted_dict_json:
            for json in v:
                output_json.append(json)
        return output_json,single_scene_index
            
    # simulation_new_new_json,single_scene_index=sort_and_batch(simulation_new_json, args.num_scenes_per_batch)
    simulation_new_new_json=simulation_new_json
    single_scene_index=0

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.planner_ckpt_dir is not None:
        cfg.ckpt.planner.ckpt_dir = args.planner_ckpt_dir
        cfg.ckpt.planner.ckpt_key = args.planner_ckpt_key

    if args.predictor_ckpt_dir is not None:
        cfg.ckpt.predictor.ckpt_dir = args.predictor_ckpt_dir
        cfg.ckpt.predictor.ckpt_key = args.predictor_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
        cfg.trajdata_data_dirs["nusc_trainvalval"] = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    
    # add eval_class into the results_dir
    # cfg.results_dir = os.path.join(cfg.results_dir, cfg.eval_class)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]
    if "ui" in cfg.edits.editing_source:
        # can only handle one scene with UI
        cfg.num_scenes_per_batch = 1

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]
    if args.trajdata_source_test=='val':
        cfg.trajdata_source_test = ["nusc_trainval-val"]
    elif args.trajdata_source_test=='train':
        cfg.trajdata_source_test = ["nusc_trainval-train"]
    cfg.pop("nusc")
    cfg.pop("trajdata")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)

    cfg.trajectory_output = args.trajectory_output
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }

    cfg.lock()
    print(cfg)
    run_scene_editor(
        device,
        cfg,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
        simulation_json=simulation_json,
        simulation_new_new_json=simulation_new_new_json,
        single_scene_index=single_scene_index
    )
