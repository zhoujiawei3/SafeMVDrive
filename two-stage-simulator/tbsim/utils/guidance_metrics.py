
from collections import defaultdict
from turtle import speed

import torch
import numpy as np

import pandas as pd

import scipy
from scipy.spatial.distance import pdist

from tbsim.utils.geometry_utils import transform_points_tensor, transform_yaw, detect_collision, CollisionType, batch_nd_transform_points_np, get_box_world_coords

from tbsim.envs.env_metrics import (
    EnvMetrics,
    split_agents_by_scene,
    agent_index_by_scene,
    step_aggregate_per_scene,
    masked_average_per_episode
)
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.metrics as Metrics

from tbsim.utils.trajdata_utils import get_raster_pix2m

############## GUIDANCE METRICS ########################

#
# NOTE: each metric should handle ONLY THE SPECIFIED SCENE and AGENTS
#       specified by the guidance configuration
#

class GuidanceMetric(EnvMetrics):
    def __init__(self, scene_idx, params, agents, num_scenes):
        '''
        - scene_idx : which scene index this guidance is used for (within each batch, i.e. the LOCAL scene index)
        - params : param dict that defines the guidance behavior
        - agents : which agents in the scene the guidance was used on.
        - num_scenes : number of total scenes in a batch (i.e. how many will be in state_info when passed in)
        '''
        self._df = None
        self._scene_ts = defaultdict(lambda:0)

        self.scene_idx = scene_idx
        self.params = params
        self.agents = agents
        self.num_scenes = num_scenes

        self.reset()

        self.global_t = 0

    def update_global_t(self, global_t=None):
        '''
        Update any persistant state needed by guidance loss functions.
        - global_t : the current global timestep of rollout
        '''
        if global_t is not None:
            self.global_t = global_t
            
class TargetSpeedGuidance(GuidanceMetric):
    """Compute the average speed satisfaction of the agents"""
    def reset(self):
        self.target_speed = self.params['target_speed']
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average speed"""
        speed = np.array(state_info['curr_speed'])
        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx
        speed = speed[scene_mask]
        if self.agents is not None:
            speed = speed[self.agents]

        # print('guidance metric self.global_t', self.global_t)
        if self.global_t < self.target_speed.shape[-1]:
            speed_deviation = np.abs(speed - self.target_speed[..., self.global_t])
        else:
            speed_deviation = np.zeros_like(speed)
        # print('speed.shape', speed.shape)
        return speed_deviation

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met = self.compute_per_step(state_info, all_scene_index)
        met = np.nanmean(met)
        # must return metric for all scenes
        all_scene_met = np.ones((len(all_scene_index))) * np.nan
        all_scene_met[self.scene_idx] = met
        self._per_step.append(all_scene_met)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scenes, num_steps]
        met = np.mean(met, axis=1)
        return met

class AgentCollisionGuidance_averaged_across_scene(GuidanceMetric):
    '''
    Similar to regular collision rate, but only operates on a single scene
    and returns the rate for the specified agents in the guidance config.
    '''
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent collision rate and type"""
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        yaw_cur_scene = state_info["yaw"][scene_mask]
        extent_cur_scene = state_info["extent"][..., :2][scene_mask]

        num_agents = pos_cur_scene.shape[0]

        coll_rates = np.zeros(num_agents)

        # compute collision rate
        for j in range(num_agents):
            other_agent_mask = np.arange(num_agents) != j
            coll = detect_collision(
                ego_pos=pos_cur_scene[j],
                ego_yaw=yaw_cur_scene[j],
                ego_extent=extent_cur_scene[j],
                other_pos=pos_cur_scene[other_agent_mask],
                other_yaw=yaw_cur_scene[other_agent_mask],
                other_extent=extent_cur_scene[other_agent_mask]
            )
            if coll is not None:
                # exclude specified agents
                if 'excluded_agents' in self.params and self.params['excluded_agents'] is not None and j in self.params['excluded_agents'] and coll[1] in self.params['excluded_agents']:
                    # print('exclude collision with agent', j, 'and agent', coll[1])
                    continue
                else:
                    coll_rates[j] = 1.

        if self.agents is not None:
            # mask to only the relevant agents
            coll_rates = coll_rates[self.agents]
        # compute per-scene collision counts (for visualization purposes)
        coll_counts = np.sum(coll_rates)

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met
class AgentCollisionGuidance_averaged_across_frame(AgentCollisionGuidance_averaged_across_scene):
    def get_episode_metrics(self):
        # met_all_steps = np.stack(self._per_step, axis=1)
        # met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        # all_scene_met = np.ones((self.num_scenes))*np.nan
        # all_scene_met[self.scene_idx] = met_guide_scene
        # return all_scene_met
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class twoAgentCollisionTypeGuidance(GuidanceMetric):

    def __init__(self, scene_idx, params, agents, num_scenes):
        self.ego_agent=params['target_ind']
        self.adversary_agent=params['ref_ind']
        self.collisionTypeDict=['A vehicle cuts in and collides with the ego vehicle','A vehicle rear-ends the ego vehicle','Ego vehicle rear-ends another vehicle','A vehicle has a head-on collision with the ego vehicle','A vehicle has a T-bone collision with the ego vehicle']
        super().__init__(scene_idx, params, agents, num_scenes)
        self.yaw_collision_type_dict = {
            'A vehicle cuts in and collides with the ego vehicle': [(-torch.pi/3, torch.pi/3)],
            'A vehicle rear-ends the ego vehicle': [(-torch.pi/3,torch.pi/3)],
            'Ego vehicle rear-ends another vehicle': [(-torch.pi/3,torch.pi/3)],
            'A vehicle has a head-on collision with the ego vehicle': [(-torch.pi,-2*torch.pi/3),(2*torch.pi/3,torch.pi)],
            'A vehicle has a T-bone collision with the ego vehicle': [(torch.pi/6,5*torch.pi/6), (-5*torch.pi/6,-torch.pi/6)]
        }
        self.ego_position_collision_type_dict = {
            'A vehicle cuts in and collides with the ego vehicle': [2],
            'A vehicle rear-ends the ego vehicle': [1],
            'Ego vehicle rear-ends another vehicle': [0],
            'A vehicle has a head-on collision with the ego vehicle': [0],
            'A vehicle has a T-bone collision with the ego vehicle': [2]
        }
        self.adv_position_collision_type_dict = {
            'A vehicle cuts in and collides with the ego vehicle': [0,2],
            'A vehicle rear-ends the ego vehicle': [0],
            'Ego vehicle rear-ends another vehicle': [1],
            'A vehicle has a head-on collision with the ego vehicle': [0],
            'A vehicle has a T-bone collision with the ego vehicle': [0]
        }
        
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        """Compute per-agent collision rate and type"""
        scenes=kwargs['scenes']
        scene=scenes[self.scene_idx]
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        yaw_cur_scene = state_info["yaw"][scene_mask]
        extent_cur_scene = state_info["extent"][..., :2][scene_mask]

        # compute collision rate
        num_agents = pos_cur_scene.shape[0]
        adv_agent_mask = np.arange(num_agents) == self.adversary_agent
        ego_agent_mask = np.arange(num_agents) == self.ego_agent

        other_agent_mask = (np.arange(num_agents) != self.ego_agent) & (np.arange(num_agents) != self.adversary_agent)
        # except_ego_agent_mask = np.arange(num_agents) != self.ego_agent
        coll_rate=np.zeros(num_agents)
        coll_ego = detect_collision(
            ego_pos=pos_cur_scene[self.ego_agent],
            ego_yaw=yaw_cur_scene[self.ego_agent],
            ego_extent=extent_cur_scene[self.ego_agent],
            other_pos=pos_cur_scene[adv_agent_mask],
            other_yaw=yaw_cur_scene[adv_agent_mask],
            other_extent=extent_cur_scene[adv_agent_mask]
        )
        coll_adv_and_other=detect_collision(
            ego_pos=pos_cur_scene[self.adversary_agent],
            ego_yaw=yaw_cur_scene[self.adversary_agent],
            ego_extent=extent_cur_scene[self.adversary_agent],
            other_pos=pos_cur_scene[other_agent_mask],
            other_yaw=yaw_cur_scene[other_agent_mask],
            other_extent=extent_cur_scene[other_agent_mask]
        )
        # coll_ego_and_other=detect_collision(
        #     ego_pos=pos_cur_scene[self.ego_agent],
        #     ego_yaw=yaw_cur_scene[self.ego_agent],
        #     ego_extent=extent_cur_scene[self.ego_agent],
        #     other_pos=pos_cur_scene[except_ego_agent_mask],
        #     other_yaw=yaw_cur_scene[except_ego_agent_mask],
        #     other_extent=extent_cur_scene[other_agent_mask]
        # )
        if coll_adv_and_other is not None and kwargs['scenes'][self.scene_idx]['right']==0:
            kwargs['scenes'][self.scene_idx]['right']=-1
            print("wrong_collide with other vehicle")
        
        relative_yaw=(yaw_cur_scene[self.ego_agent]-yaw_cur_scene[self.adversary_agent]+ torch.pi)% (2 * torch.pi) - torch.pi
        if coll_ego is not None:
            if len(kwargs["guidance_config"][0])>=3 and kwargs["guidance_config"][0][2]['name']=='collisiondiskLoss':
                print("name",kwargs["guidance_config"][0][2]['name'])
                kwargs["guidance_config"][0].pop(2)
            # exclude specified agents
            coll_adv = detect_collision(
                ego_pos=pos_cur_scene[self.adversary_agent],
                ego_yaw=yaw_cur_scene[self.adversary_agent],
                ego_extent=extent_cur_scene[self.adversary_agent],
                other_pos=pos_cur_scene[ego_agent_mask],
                other_yaw=yaw_cur_scene[ego_agent_mask],
                other_extent=extent_cur_scene[ego_agent_mask]
            )
            coll_rate[self.ego_agent]=1
            if kwargs['scenes'][self.scene_idx]['right']==0:
            #     collisiontype=self.collisiontype    
            #     collision_position_type=coll[0]
            #     position_allow_list=self.position_collision_type_dict[collisiontype]
            #     yaw_allow_list=self.yaw_collision_type_dict[collisiontype]
            #     position_allow=False
            #     for position_type in position_allow_list:
            #         if collision_position_type==position_type:
            #             position_allow=True
            #             break
            #     yaw_allow=False
            #     for yaw_type in yaw_allow_list:
            #         if relative_yaw>=yaw_type[0] and relative_yaw<=yaw_type[1]:
            #             yaw_allow=True
            #             break
            #     if position_allow and yaw_allow:
            #         kwargs['scenes'][self.scene_idx]['right']=1
            #         kwargs['scenes'][self.scene_idx]['generate_collision_type']=collisiontype
            #         print("right_collide")
            #         print("collision_type:",collisiontype)
            #     else:
            #         kwargs['scenes'][self.scene_idx]['right']=-1
            #         print("wrong_collide")
                for collision_type in self.collisionTypeDict:
                    ego_collision_position_type=coll_ego[0]
                    adv_collision_position_type=coll_adv[0]
                    ego_position_allow_list=self.ego_position_collision_type_dict[collision_type]
                    adv_position_allow_list=self.adv_position_collision_type_dict[collision_type]
                    yaw_allow_list=self.yaw_collision_type_dict[collision_type]
                    ego_position_allow=False
                    adv_position_allow=False
                    yaw_allow=False
                    for position_type in ego_position_allow_list:
                        if ego_collision_position_type==position_type:
                            ego_position_allow=True
                            break
                    for position_type in adv_position_allow_list:
                        if adv_collision_position_type==position_type:
                            adv_position_allow=True
                            break
                    
                    for yaw_type in yaw_allow_list:
                        if relative_yaw>=yaw_type[0] and relative_yaw<=yaw_type[1]:
                            yaw_allow=True
                            break
                    if ego_position_allow and yaw_allow and adv_position_allow:
                        kwargs['scenes'][self.scene_idx]['right']=1
                        kwargs['scenes'][self.scene_idx]['generate_collision_type']=collision_type
                        print("collide")
                        print("collision_type:",collision_type)
                if kwargs['scenes'][self.scene_idx]['right']==0:
                    kwargs['scenes'][self.scene_idx]['right']=2
                    print("collide,but no collide type exist!")
                    
        return coll_rate

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met_all = self.compute_per_step(state_info, all_scene_index,**kwargs)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met
class EgoEscapeTypeGuidance(GuidanceMetric):

    def __init__(self, scene_idx, params, agents, num_scenes):
        self.ego_agent=0
        self.adversary_agent=params['ref_ind']
        
        
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        """Compute per-agent collision rate and type"""
        scenes=kwargs['scenes']
        scene=scenes[0]
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == 0
        pos_cur_scene = state_info["centroid"][scene_mask]
        yaw_cur_scene = state_info["yaw"][scene_mask]
        extent_cur_scene = state_info["extent"][..., :2][scene_mask]

        # compute collision rate
        num_agents = pos_cur_scene.shape[0]
        ego_agent_mask = np.arange(num_agents) == self.ego_agent
        except_ego_agent_mask = np.arange(num_agents) != self.ego_agent
        coll_rate=np.zeros(num_agents)
        coll_ego_and_other=detect_collision(
            ego_pos=pos_cur_scene[self.ego_agent],
            ego_yaw=yaw_cur_scene[self.ego_agent],
            ego_extent=extent_cur_scene[self.ego_agent],
            other_pos=pos_cur_scene[except_ego_agent_mask],
            other_yaw=yaw_cur_scene[except_ego_agent_mask],
            other_extent=extent_cur_scene[except_ego_agent_mask]
        )
        print("second_time_right",kwargs['scenes'][0]['second_time_right'])
        if coll_ego_and_other is not None and kwargs['scenes'][0]['second_time_right']==0:
            kwargs['scenes'][0]['second_time_right']=-1
            coll_rate[self.ego_agent]=1
            print("collide with other vehicle")    
        return coll_rate

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met_all = self.compute_per_step(state_info, all_scene_index,**kwargs)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((1))*np.nan
        all_scene_met[0] = met_guide_scene
        return all_scene_met
class MinDistanceGuidance(GuidanceMetric):
    """
    Guidance metric that tracks and returns the minimum distance between two specified agents
    throughout a scenario.
    """
    def __init__(self, scene_idx, params, agents, num_scenes):
        """
        - scene_idx: which scene index this guidance is used for (within each batch)
        - params: param dict containing target_ind and ref_ind (the two agents to measure distance between)
        - agents: which agents in the scene the guidance was used on
        - num_scenes: number of total scenes in a batch
        """
        self.target_ind = 0  # First agent (typically ego)
        self.ref_ind = params['ref_ind']        # Second agent (typically reference/adversary)
        super().__init__(scene_idx, params, agents, num_scenes)
        
    def reset(self):
        """Reset metric tracking for a new episode"""
        self._per_step = []
        self._min_distance = float('inf')  # Track minimum distance across episode
        
    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray, **kwargs):
        """Compute distance between the two specified agents for current step"""
        # Get indices for the scene we're interested in
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == 0
        
        # Extract positions for the current scene
        pos_cur_scene = state_info["centroid"][scene_mask]
        
        # Compute Euclidean distance between the two agents
        target_pos = pos_cur_scene[0]
        ref_pos = pos_cur_scene[self.ref_ind]
        
        # Calculate distance - use numpy for consistency with other metrics
        distance = np.linalg.norm(target_pos - ref_pos)
        
        # Update minimum distance if current distance is smaller
        if distance < self._min_distance:
            self._min_distance = distance
            
        return distance
    
    def add_step(self, state_info: dict, all_scene_index: np.ndarray, **kwargs):
        """Add current step's distance to tracking"""
        distance = self.compute_per_step(state_info, all_scene_index, **kwargs)
        self._per_step.append(distance)
        
        # Store the distance in the scene info if available
        if 'scenes' in kwargs and kwargs['scenes'] is not None:
            scene = kwargs['scenes'][0]
            if 'min_distance' not in scene or distance < scene['min_distance']:
                kwargs['scenes'][0]['min_distance'] = float(distance)
                print("distance",kwargs['scenes'][0]['min_distance'])
                
    
    def get_episode_metrics(self):
        """Return the minimum distance over the episode for the specified scene"""
        # Create result array with NaN for all scenes except the one we're tracking
        all_scene_met = np.ones((1)) * np.nan
        
        # For our scene, return the minimum distance
        if len(self._per_step) > 0:
            min_distance = min(self._per_step)
            all_scene_met[0] = min_distance
        else:
            all_scene_met[0] = float('inf')
            
        return all_scene_met
class AgentCollisionGuidanceDisk_averaged_across_scene(GuidanceMetric):
    '''
    Similar to regular metric, but only counts collision if a single disk that approximates 
    each agent overlaps (including the BUFFER DIST if desired).
    '''
    def __init__(self, scene_idx, params, agents, num_scenes, use_buffer_dist=False):
        super().__init__(scene_idx, params, agents, num_scenes)
        self.use_buffer_dist = use_buffer_dist

    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent collision rate and type"""
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        extent_cur_scene = state_info["extent"][..., :2][scene_mask]
        rad_cur_scene = np.amin(extent_cur_scene, axis=-1) / 2.0

        num_agents = pos_cur_scene.shape[0]

        coll_rates = np.zeros(num_agents)

        # compute collision rate
        for j in range(num_agents):
            # computes if j collides with ANY other agent. only counts single collision
            #       to be consistent with other coll metric.
            other_agent_mask = np.arange(num_agents) != j
            if 'excluded_agents' in self.params and self.params['excluded_agents'] is not None:
                other_agent_mask[self.params['excluded_agents']] = False
            neighbor_dist = np.linalg.norm(pos_cur_scene[j:j+1] - pos_cur_scene[other_agent_mask], axis=-1)
            min_allowed_dist = rad_cur_scene[j] + rad_cur_scene[other_agent_mask]
            if self.use_buffer_dist:
                min_allowed_dist = min_allowed_dist + self.params['buffer_dist']
            coll = np.sum(neighbor_dist < min_allowed_dist) > 0
            if coll:
                coll_rates[j] = 1.

        # exclude specified agents
        if 'excluded_agents' in self.params and self.params['excluded_agents'] is not None:
            coll_rates[self.params['excluded_agents']] = 0

        if self.agents is not None:
            # mask to only the relevant agents
            coll_rates = coll_rates[self.agents]
        # compute per-scene collision counts (for visualization purposes)
        coll_counts = np.sum(coll_rates)

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class AgentCollisionGuidanceDisk_averaged_across_frame(AgentCollisionGuidanceDisk_averaged_across_scene):
    def get_episode_metrics(self):
        # met_all_steps = np.stack(self._per_step, axis=1)
        # met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        # all_scene_met = np.ones((self.num_scenes))*np.nan
        # all_scene_met[self.scene_idx] = met_guide_scene
        # return all_scene_met
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class SocialDistanceGuidance_averaged_across_scene(AgentCollisionGuidanceDisk_averaged_across_scene):
    def __init__(self, scene_idx, params, agents, num_scenes):
        super().__init__(scene_idx, params, agents, num_scenes, use_buffer_dist=True)

class SocialDistanceGuidance_averaged_across_frame(SocialDistanceGuidance_averaged_across_scene):
    def get_episode_metrics(self):
        # met_all_steps = np.stack(self._per_step, axis=1)
        # met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        # all_scene_met = np.ones((self.num_scenes))*np.nan
        # all_scene_met[self.scene_idx] = met_guide_scene
        # return all_scene_met
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class TargetPosGuidance(GuidanceMetric):
    """
    How well target pos (at any timestep) are met.
    """
    def reset(self):
        self.locs = np.array(self.params['target_pos'])
        self._agent_from_world = None
        self._cur_min_dist = np.ones((self.locs.shape[0]))*np.inf

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self._agent_from_world is None:
            # only want to record this at the first step (the planning step)
            #       b/c this is what the target is in.
            #       TODO: when the target is global, need to change this
            self._agent_from_world = state_info["agent_from_world"][scene_mask]
        pos_cur_scene = batch_nd_transform_points_np(pos_cur_scene[:,None], self._agent_from_world)[:,0]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        cur_dist = np.linalg.norm(pos_cur_scene - self.locs, axis=-1)
        new_min_mask = cur_dist < self._cur_min_dist
        self._cur_min_dist[new_min_mask] = cur_dist[new_min_mask]

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        met = np.mean(self._cur_min_dist)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class GlobalTargetPosGuidance(GuidanceMetric):
    """
    How well global target pos (at any timestep) are met.
    """
    def reset(self):
        self.locs = np.array(self.params['target_pos'])
        self._cur_min_dist = np.ones((self.locs.shape[0]))*np.inf

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        cur_dist = np.linalg.norm(pos_cur_scene - self.locs, axis=-1)
        new_min_mask = cur_dist < self._cur_min_dist
        self._cur_min_dist[new_min_mask] = cur_dist[new_min_mask]

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        met = np.mean(self._cur_min_dist)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class ConstraintGuidance(GuidanceMetric):
    """
    How well constraint (waypoints at specific time) are met.
    """
    def reset(self):
        if 'locs' in self.params and 'times' in self.params:
            # true hard constraint
            self.locs = np.array(self.params['locs'])
            self.times = np.array(self.params['times'])
        elif 'target_pos' in self.params and 'target_time' in self.params:
            # guidance version
            self.locs = np.array(self.params['target_pos'])
            self.times = np.array(self.params['target_time'])
        else:
            raise NotImplementedError()
        # NOTE: assumes add_step will be called for initial state before model prediction as well
        self.times = self.times + 1
        self._agent_from_world = None
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self._agent_from_world is None:
            # only want to record this at the first step (the planning step)
            #       b/c this is what the target is in.
            #       TODO: when the target is global, need to change this
            self._agent_from_world = state_info["agent_from_world"][scene_mask]
        pos_cur_scene = batch_nd_transform_points_np(pos_cur_scene[:,None], self._agent_from_world)[:,0]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        return pos_cur_scene

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        step_pos = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(step_pos)

    def get_episode_metrics(self):
        ''' finally see if were met at desired step'''
        all_pos = np.stack(self._per_step, axis=1)  # [num_agents, num_steps, 2]
        tgt_pos = all_pos[np.arange(all_pos.shape[0]),self.times]
        met = np.mean(np.linalg.norm(tgt_pos - self.locs, axis=-1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class GlobalConstraintGuidance(GuidanceMetric):
    """
    How well constraint (waypoints at specific time) are met.
    """
    def reset(self):
        if 'locs' in self.params and 'times' in self.params:
            # true hard constraint
            self.locs = np.array(self.params['locs'])
            self.times = np.array(self.params['times'])
        elif 'target_pos' in self.params and 'target_time' in self.params:
            # guidance version
            self.locs = np.array(self.params['target_pos'])
            self.times = np.array(self.params['target_time'])
        else:
            raise NotImplementedError()
        # NOTE: assumes add_step will be called for initial state before model prediction as well
        self.times = self.times + 1
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]
        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]
        return pos_cur_scene

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        step_pos = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(step_pos)

    def get_episode_metrics(self):
        ''' finally see if were met at desired step'''
        all_pos = np.stack(self._per_step, axis=1)  # [num_agents, num_steps, 2]
        tgt_pos = all_pos[np.arange(all_pos.shape[0]),self.times]
        met = np.mean(np.linalg.norm(tgt_pos - self.locs, axis=-1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class MapCollisionGuidance_averaged_across_frame(GuidanceMetric):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        scenes=kwargs['scenes']
        scene=scenes[self.scene_idx]
        self.ego_agent=0
        self.adversary_agent=scene['ref_ind']
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        extent = obs["extent"][:,:2]
        # TODO: this is super hacky and assumes trajdata is being used.
        #       should really just transform corners in world frame then convert to raster.
        extent = get_raster_pix2m()*extent # convert to raster frame

        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx

        drivable_region = drivable_region[scene_mask]
        centroid_raster = centroid_raster[scene_mask]
        cur_yaw = cur_yaw[scene_mask]
        extent = extent[scene_mask]
        if self.agents is not None:
            drivable_region = drivable_region[self.agents]
            centroid_raster = centroid_raster[self.agents]
            cur_yaw = cur_yaw[self.agents]
            extent = extent[self.agents]

        # from matplotlib import pyplot as plt
        # boxes = get_box_world_coords(centroid_raster, cur_yaw, extent)  # [B, ..., 4, 2]
        # print(boxes.size())
        # i2pl = 1
        # cur_layer = drivable_region[i2pl].cpu().numpy() # h, w
        # fig = plt.figure()
        # plt.imshow(cur_layer)
        # plt.plot(centroid_raster[i2pl, 0:1].cpu().numpy(), centroid_raster[i2pl, 1:2].cpu().numpy(), 'go')
        # for ci in range(4):
        #     plt.plot(boxes[i2pl, ci:ci+1, 0].cpu().numpy(), boxes[i2pl, ci:ci+1, 1].cpu().numpy(), 'ro')
        # plt.savefig('map_collision_metric.png')
        # plt.close(fig)
        # raise

        off_road = Metrics.batch_detect_off_road_boxes(centroid_raster, cur_yaw, extent, drivable_region)
        # print(off_road)
        if off_road[self.adversary_agent]==1:
            if scene['right']==0:
                scene['right']=-1
                print("wrong, off road")
        if off_road[self.ego_agent]==1 and "second_time_right" in scene:
            if scene['second_time_right']==0:
                scene['second_time_right']=-1
                print("wrong, off road")
        off_road = TensorUtils.to_numpy(off_road)

        return off_road

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met = self.compute_per_step(state_info, all_scene_index,**kwargs)
        self._per_step.append(met)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class MapCollisionGuidanceDisk_averaged_across_frame(GuidanceMetric):
    """Compute the fraction of the time that the agent is in undrivable regions.
            each agent is approximated by a single disk"""
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        extent = obs["extent"][:,:2]
        # TODO: this is super hacky and assumes trajdata is being used.
        #       should really just transform corners in world frame then convert to raster.
        extent = get_raster_pix2m()*extent # convert to raster frame

        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx

        drivable_region = drivable_region[scene_mask]
        centroid_raster = centroid_raster[scene_mask]
        cur_yaw = cur_yaw[scene_mask]
        extent = extent[scene_mask]
        if self.agents is not None:
            drivable_region = drivable_region[self.agents]
            centroid_raster = centroid_raster[self.agents]
            cur_yaw = cur_yaw[self.agents]
            extent = extent[self.agents]

        # import matplotlib
        # from matplotlib import pyplot as plt
        # boxes = get_box_world_coords(centroid_raster, cur_yaw, extent)  # [B, ..., 4, 2]
        # print(boxes.size())
        # i2pl = 1
        # cur_layer = drivable_region[i2pl].cpu().numpy() # h, w
        # fig = plt.figure()
        # plt.imshow(cur_layer)
        # plt.plot(centroid_raster[i2pl, 0:1].cpu().numpy(), centroid_raster[i2pl, 1:2].cpu().numpy(), 'go')
        # for ci in range(4):
        #     plt.plot(boxes[i2pl, ci:ci+1, 0].cpu().numpy(), boxes[i2pl, ci:ci+1, 1].cpu().numpy(), 'ro')
        # plt.show()
        # plt.close(fig)

        off_road = Metrics.batch_detect_off_road_disk(centroid_raster, extent, drivable_region)

        # print(off_road)
        off_road = TensorUtils.to_numpy(off_road)
        
        return off_road

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met = self.compute_per_step(state_info, all_scene_index,**kwargs)
        self._per_step.append(met)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        # fraction of frames colliding
        # print(np.sum(met_all_steps, axis=1))
        met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        met_guide_scene = np.mean(met_guide_scene)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

class MapCollisionGuidance_averaged_across_scene(MapCollisionGuidance_averaged_across_frame):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def get_episode_metrics(self):
        
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met
        # met_all_steps = np.stack(self._per_step, axis=1)
        # # fraction of frames colliding
        # # print(np.sum(met_all_steps, axis=1))
        # met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        # met_guide_scene = np.mean(met_guide_scene)
        # all_scene_met = np.ones((self.num_scenes))*np.nan
        # all_scene_met[self.scene_idx] = met_guide_scene
        # return all_scene_met

class MapCollisionGuidanceDisk_averaged_across_scene(MapCollisionGuidanceDisk_averaged_across_frame):

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met
        # met_all_steps = np.stack(self._per_step, axis=1)
        # # fraction of frames colliding
        # # print(np.sum(met_all_steps, axis=1))
        # met_guide_scene = np.sum(met_all_steps, axis=1) / float(met_all_steps.shape[1])
        # met_guide_scene = np.mean(met_guide_scene)
        # all_scene_met = np.ones((self.num_scenes))*np.nan
        # all_scene_met[self.scene_idx] = met_guide_scene
        # return all_scene_met

class SocialGroupGuidance(GuidanceMetric):
    """
    How well social group stays together.
    At each step computes how well min distance to another agent in the social group
    matches the desired social_dist.
    """
    def reset(self):
        self.social_dist = self.params['social_dist']
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        ''' stores all given positions for needed agents '''
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]

        if self.agents is not None:
            pos_cur_scene = pos_cur_scene[self.agents]

        B = pos_cur_scene.shape[0]
        pairwise_dist = np.linalg.norm(pos_cur_scene[np.newaxis] - pos_cur_scene[:,np.newaxis], axis=-1)
        pairwise_dist[np.arange(B),np.arange(B)] = np.inf
        min_dist = np.amin(pairwise_dist, axis=-1)
        step_err = np.abs(min_dist - self.social_dist)
        self._per_step.append(step_err)

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        all_err = np.stack(self._per_step, axis=1)  # [num_agents, num_steps]
        met = np.mean(all_err)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met




############## TBD ########################
class StopSignGuidance(GuidanceMetric):
    """
    Compute the stop sign guidance metric.
    Consider the stop sign region to be in local frame at the first rollout step.
    """
    def reset(self):
        self.stop_sign = None
        self._speed_in_stop_region = None
        self._agent_from_world = None

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average speed"""
        
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx

        agent_hist = state_info['agent_hist'][scene_mask]
        agent_from_world = state_info["agent_from_world"][scene_mask]
        world_from_agent = state_info["world_from_agent"][scene_mask]
        if self.agents is not None:
            agent_hist = agent_hist[self.agents]
            world_from_agent = world_from_agent[self.agents]
            agent_from_world = agent_from_world[self.agents]

        # set this only once at the first rollout step
        if self._agent_from_world is None:
            self._agent_from_world = agent_from_world

        # in agent coordinates
        pos_hist = agent_hist[...,-1:,:2]
        vx_hist = agent_hist[...,-1:,2]
        vy_hist = agent_hist[...,-1:,3]
        speed_hist = np.sqrt(vx_hist**2 + vy_hist**2)

        # convert from agent to world frame
        pos_hist = batch_nd_transform_points_np(pos_hist, world_from_agent)
        # convert from world frame to first local frame
        pos_hist = batch_nd_transform_points_np(pos_hist, self._agent_from_world)

        # only set this once
        if self.stop_sign is None:        
            from tbsim.rules.stl_traffic_rules import StopSignRule
            self.stop_sign_pos = np.array(self.params['stop_sign_pos'])
            self.stop_box_dim = np.array(self.params['stop_box_dim'])

            if self.agents is not None:
                assert self.stop_sign_pos.shape[0] == self.stop_box_dim.shape[0] == len(self.agents)
            
            self.stop_sign = StopSignRule(self.stop_sign_pos, self.stop_box_dim, self.params['low_speed_th'])
        
        # [B, T]
        inclusion_mask = self.stop_sign.check_inclusion(pos_hist[:, :, None, :]).reshape([pos_hist.shape[0], pos_hist.shape[1]])

        inclusion_speed_const = np.ones_like(inclusion_mask)*np.inf

        inclusion_speed_const[inclusion_mask==1] = 0
        speed_hist = speed_hist + inclusion_speed_const

        if self._speed_in_stop_region is None:
            self._speed_in_stop_region = speed_hist
        else:
            self._speed_in_stop_region = np.concatenate((self._speed_in_stop_region, speed_hist), axis=1)

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        rollout_min_speed = np.sort(self._speed_in_stop_region, axis=1)[:, :self.params['num_time_steps_to_stop']]

        # print('self._speed_in_stop_region.shape', self._speed_in_stop_region.shape)
        # print('rollout_min_speed.shape', rollout_min_speed.shape)
        # print('rollout_min_speed', rollout_min_speed[:, :10])

        min_speed_list = []
        for i in range(rollout_min_speed.shape[0]):
            min_speed_i = rollout_min_speed[i]
            if min_speed_i[0] == np.inf:
                continue
            else:
                mean_min_speed_i = np.mean(min_speed_i[~np.isinf(min_speed_i)])
                min_speed_list.append(mean_min_speed_i)
        
        met = np.mean(min_speed_list)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

# TBD: add a longer agent history field in state_info such that we can estimate past stop sign behavior for a larger action_num
class GlobalStopSignGuidance(GuidanceMetric):
    """
    Compute the global stop sign guidance metric.
    Consider the stop sign region to be in world frame.
    """
    def reset(self):
        self.stop_sign = None
        self._speed_in_stop_region = None

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average speed"""
        
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx

        agent_hist = state_info['agent_hist'][scene_mask]
        world_from_agent = state_info["world_from_agent"][scene_mask]
        if self.agents is not None:
            agent_hist = agent_hist[self.agents]
            world_from_agent = world_from_agent[self.agents]

        # in agent coordinates
        pos_hist = agent_hist[...,-1:,:2]
        vx_hist = agent_hist[...,-1:,2]
        vy_hist = agent_hist[...,-1:,3]
        speed_hist = np.sqrt(vx_hist**2 + vy_hist**2)

        # convert from agent to world frame
        pos_hist = batch_nd_transform_points_np(pos_hist, world_from_agent)

        if self.stop_sign is None:        
            from tbsim.rules.stl_traffic_rules import StopSignRule
            self.stop_sign_pos = np.array(self.params['stop_sign_pos'])
            self.stop_box_dim = np.array(self.params['stop_box_dim'])

            if self.agents is not None:
                assert self.stop_sign_pos.shape[0] == self.stop_box_dim.shape[0] == len(self.agents)
            
            self.stop_sign = StopSignRule(self.stop_sign_pos, self.stop_box_dim, self.params['low_speed_th'])
        
        # [B, T]
        inclusion_mask = self.stop_sign.check_inclusion(pos_hist[:, :, None, :]).reshape([pos_hist.shape[0], pos_hist.shape[1]])

        inclusion_speed_const = np.ones_like(inclusion_mask)*np.inf
        inclusion_speed_const[inclusion_mask==1] = 0
        speed_hist = speed_hist + inclusion_speed_const

        if self._speed_in_stop_region is None:
            self._speed_in_stop_region = speed_hist
        else:
            self._speed_in_stop_region = np.concatenate((self._speed_in_stop_region, speed_hist), axis=1)

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        self.compute_per_step(state_info, all_scene_index)

    def get_episode_metrics(self):
        rollout_min_speed = np.sort(self._speed_in_stop_region, axis=1)[:, :self.params['num_time_steps_to_stop']]

        # print('self._speed_in_stop_region.shape', self._speed_in_stop_region.shape)
        # print('rollout_min_speed.shape', rollout_min_speed.shape)
        # print('rollout_min_speed', rollout_min_speed[:, :10])
        
        min_speed_list = []
        for i in range(rollout_min_speed.shape[0]):
            min_speed_i = rollout_min_speed[i]
            if min_speed_i[0] == np.inf:
                continue
            else:
                mean_min_speed_i = np.mean(min_speed_i[~np.isinf(min_speed_i)])
                min_speed_list.append(mean_min_speed_i)
        
        met = np.mean(min_speed_list)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met
        return all_scene_met

class SpeedLimitGuidance(GuidanceMetric):
    """Compute the average speed satisfaction of the agents"""
    def reset(self):
        self.speed_limit = self.params['speed_limit']
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average speed violation"""
        speed = np.array(state_info['curr_speed'])
        # filter for specified scene and agents only
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx
        speed = speed[scene_mask]
        if self.agents is not None:
            speed = speed[self.agents]
        speed_violation = speed - self.speed_limit
        speed_violation = np.clip(speed_violation, a_min=0, a_max=None)
        return speed_violation

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met = self.compute_per_step(state_info, all_scene_index)
        met = np.mean(met)
        # must return metric for all scenes
        all_scene_met = np.ones((len(all_scene_index)))*np.nan
        all_scene_met[self.scene_idx] = met
        self._per_step.append(all_scene_met)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scenes, num_steps]
        met = np.mean(met, axis=1)
        return met

class AccLimitGuidance(GuidanceMetric):
    """Compute the average speed satisfaction of the agents"""
    def reset(self):
        self.acc_limit = self.params['acc_limit']
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average acc violation"""
        # which agent indices belong to each scene (list of np arrays with agent inds)
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()

        scene_mask = local_scene_index == self.scene_idx

        agent_hist = state_info['agent_hist'][scene_mask]
        if self.agents is not None:
            agent_hist = agent_hist[self.agents]

        # in agent coordinates
        lon_acc = agent_hist[...,-1:,4]
        cos_psi = agent_hist[...,-1:,7]
        acc = lon_acc / cos_psi 

        acc_dev = np.abs(acc) - self.acc_limit
        acc_dev = np.clip(acc_dev, a_min=0, a_max=None)
        avg_acc_violation = np.mean(acc_dev)

        return avg_acc_violation

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met = self.compute_per_step(state_info, all_scene_index)
        met = np.mean(met)
        # must return metric for all scenes
        all_scene_met = np.ones((len(all_scene_index)))*np.nan
        all_scene_met[self.scene_idx] = met
        self._per_step.append(all_scene_met)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scenes, num_steps]
        met = np.mean(met, axis=1)
        return met

class GPTGuidance(GuidanceMetric):
    """Compute the average metric satisfaction of the agents"""
    def reset(self):
        self._per_step = []

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene average metric"""
        return 0

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met = self.compute_per_step(state_info, all_scene_index)
        met = np.mean(met)
        # must return metric for all scenes
        all_scene_met = np.ones((len(all_scene_index)))*np.nan
        all_scene_met[self.scene_idx] = met
        self._per_step.append(all_scene_met)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scenes, num_steps]
        met = np.mean(met, axis=1)
        return met

class GPTCollisionGuidance(AgentCollisionGuidance_averaged_across_scene):
    def __init__(self, scene_idx, params, agents, num_scenes):
        agents = [params['target_ind'], params['ref_ind']]
        # we only count the loss on one pair of agents for each scene
        super().__init__(scene_idx, params, agents, num_scenes)
        
    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = 1 - met_guide_scene
        return all_scene_met


class CollisionTypeGuidance(AgentCollisionGuidance_averaged_across_scene):
    def __init__(self, scene_idx, params, agents, num_scenes):
        agents = [params['target_ind'], params['ref_ind']]
        super().__init__(scene_idx, params, agents, num_scenes)
        # we only count the loss on one pair of agents for each scene

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = 1 - met_guide_scene
        return all_scene_met
class CollisionDiskGuidance(AgentCollisionGuidanceDisk_averaged_across_scene):
    def __init__(self, scene_idx, params, agents, num_scenes):
        agents = [params['target_ind'], params['ref_ind']]
        super().__init__(scene_idx, params, agents, num_scenes)
        # we only count the loss on one pair of agents for each scene

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1)
        met_guide_scene = np.mean(np.amax(met_all_steps, axis=1))
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = 1 - met_guide_scene
        return all_scene_met
class GPTKeepDistanceLoss(GuidanceMetric):
    def __init__(self, scene_idx, params, agents, num_scenes):
        super().__init__(scene_idx, params, agents, num_scenes)
        # we only count the loss on one pair of agents for each scene

    def reset(self):
        self._per_step = []
        self.min_distance = self.params['min_distance']
        self.max_distance = self.params['max_distance']
        self.target_ind = self.params['target_ind']
        self.ref_ind = self.params['ref_ind']

    def compute_per_step(self, state_info: dict, all_scene_index: np.ndarray):
        local_scene_index = torch.unique_consecutive(torch.tensor(state_info["scene_split"]), return_inverse=True)[1].numpy()
        scene_mask = local_scene_index == self.scene_idx
        pos_cur_scene = state_info["centroid"][scene_mask]

        neighbor_dist = np.linalg.norm(pos_cur_scene[self.target_ind] - pos_cur_scene[self.ref_ind], axis=-1)
        if neighbor_dist < self.min_distance:
            dist_diff = self.min_distance - neighbor_dist
        elif neighbor_dist > self.max_distance:
            dist_diff = neighbor_dist - self.max_distance
        else:
            dist_diff = 0

        return dist_diff

    def add_step(self, state_info: dict, all_scene_index: np.ndarray,**kwargs):
        met_all = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met_all)

    def get_episode_metrics(self):
        met_guide_scene = np.mean(self._per_step)
        all_scene_met = np.ones((self.num_scenes))*np.nan
        all_scene_met[self.scene_idx] = met_guide_scene
        return all_scene_met

    
############## GUIDANCE METRICS utilities ########################

GUIDANCE_NAME_TO_METRICS = {
    'target_speed' :              {'target_speed' : TargetSpeedGuidance},
    'agent_collision' :           {'agent_collision_disk_averaged_across_scene' : AgentCollisionGuidanceDisk_averaged_across_scene, 'social_dist_averaged_across_scene' : SocialDistanceGuidance_averaged_across_scene, 'agent_collision_averaged_across_scene' : AgentCollisionGuidance_averaged_across_scene,'agent_collision_disk_averaged_across_frame' : AgentCollisionGuidanceDisk_averaged_across_frame, 'social_dist_averaged_across_frame' : SocialDistanceGuidance_averaged_across_frame, 'agent_collision_averaged_across_frame' : AgentCollisionGuidance_averaged_across_frame},
    'map_collision' :             {'map_collision_averaged_across_frame' : MapCollisionGuidance_averaged_across_frame, 'map_collision_disk_averaged_across_frame' : MapCollisionGuidanceDisk_averaged_across_frame,'map_collision_averaged_across_scene' : MapCollisionGuidance_averaged_across_scene, 'map_collision_disk_averaged_across_scene' : MapCollisionGuidanceDisk_averaged_across_scene},
    'target_pos_at_time' :        {'target_pos_at_time' : ConstraintGuidance},
    'target_pos' :                {'target_pos' : TargetPosGuidance},
    'global_target_pos_at_time' : {'global_target_pos_at_time' : GlobalConstraintGuidance},
    'global_target_pos' :         {'global_target_pos' : GlobalTargetPosGuidance},
    'social_group' :              {'social_group' : SocialGroupGuidance},

    'stop_sign': {'stop_sign': StopSignGuidance},
    'global_stop_sign': {'global_stop_sign': GlobalStopSignGuidance},
    'speed_limit': {'speed_limit': SpeedLimitGuidance},
    'acc_limit': {'acc_limit': AccLimitGuidance},
    'gpt': {'gpt': GPTGuidance},
    'gptcollision': {'gptcollision': GPTCollisionGuidance, "twoAgentCollisionTypeGuidance": twoAgentCollisionTypeGuidance},
    'gptkeepdistance': {'gptkeepdistance': GPTKeepDistanceLoss},
    'collisiontypeLoss': {'collisiontypeLoss': CollisionTypeGuidance},
    'collisiondiskLoss': {'collisiondiskLoss': CollisionDiskGuidance, "twoAgentCollisionTypeGuidance": twoAgentCollisionTypeGuidance},
    'ego_escape_loss' : {'agent_collision_disk_averaged_across_scene' : AgentCollisionGuidanceDisk_averaged_across_scene, 'social_dist_averaged_across_scene' : SocialDistanceGuidance_averaged_across_scene, 'agent_collision_averaged_across_scene' : AgentCollisionGuidance_averaged_across_scene,'agent_collision_disk_averaged_across_frame' : AgentCollisionGuidanceDisk_averaged_across_frame, 'social_dist_averaged_across_frame' : SocialDistanceGuidance_averaged_across_frame, 'agent_collision_averaged_across_frame' : AgentCollisionGuidance_averaged_across_frame, 'ego_escape' : EgoEscapeTypeGuidance,'min_distance':MinDistanceGuidance},
}

def guidance_metrics_from_config(guidance_config):
    '''
    Returns metrics objects to measure success of each guidance.
    '''
    metrics = dict()
    num_scenes = len(guidance_config)
    for si in range(num_scenes):
        cur_cfg_list = guidance_config[si]
        for ci, cur_cfg in enumerate(cur_cfg_list):
            if cur_cfg['name'] in GUIDANCE_NAME_TO_METRICS:
                guide_mets = GUIDANCE_NAME_TO_METRICS[cur_cfg['name']]
                for met_name, met_func in guide_mets.items():
                    cur_metric = met_func(si, cur_cfg['params'], cur_cfg['agents'], num_scenes)
                    # could be multiple of the same guidance in a scene
                    metrics['guide_' + met_name + '_s%dg%d' % (si, ci)] = cur_metric
            else:
                raise ValueError('Unknown guidance metric: {}'.format(cur_cfg['name']))
            # the following might lead to the metric of the last item will always be used for the metric name which may not be desired
            # # could be multiple of the same guidance in a scene
            # metrics['guide_' + cur_cfg['name'] + '_s%dg%d' % (si, ci)] = cur_metric
    return metrics

def constraint_metrics_from_config(constraint_config):
    metrics = dict()
    num_scenes = len(constraint_config)
    for si in range(num_scenes):
        cur_cfg = constraint_config[si]
        cur_metric = ConstraintGuidance(si,
                                        {'locs' : cur_cfg['locs'], 'times' : cur_cfg['times']},
                                        cur_cfg['agents'],
                                        num_scenes
                                        )
        metrics['guide_constraint_s%d' % (si)] = cur_metric
    return metrics
