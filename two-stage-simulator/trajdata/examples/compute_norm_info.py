from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union

from trajdata import AgentBatch, SceneBatch, AgentType, UnifiedDataset
from trajdata.utils.arr_utils import angle_wrap

import numpy as np
import torch
from torch import Tensor

import os

def main(dataset_to_use, dataset_loader_to_use, centric, keys_to_compute, hist_sec = 1.0, fut_sec = 2.0, steps = None, agent_types = [AgentType.VEHICLE]):
    dt = 0.1

    interaction_d = 50 # distance to scene ego to be included. [30, 50, np.inf]
    max_agent_num = None # scene-centric
    max_neighbor_num = 20 # None # agent-centric

    if centric == 'scene':
        dataloader_batch_size = 2
    else:
        dataloader_batch_size = 50
        
    if dataset_loader_to_use == "unified":
        dataset = UnifiedDataset(
            desired_data=[dataset_to_use],
            centric=centric,
            desired_dt=dt,
            history_sec=(hist_sec, hist_sec),
            future_sec=(fut_sec, fut_sec), # This should be consistent with the horizon diffusion model uses
            only_types=agent_types,
            only_predict=agent_types,
            agent_interaction_distances=defaultdict(lambda: interaction_d),
            incl_robot_future=False,
            incl_raster_map=False,
            raster_map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
            state_format="x,y,xd,yd,xdd,ydd,h",
            obs_format="x,y,xd,yd,xdd,ydd,s,c",
            # augmentations=[noise_hists],
            data_dirs={
                "nusc_trainval": "../behavior-generation-dataset/nuscenes",
                "nusc_mini": "../behavior-generation-dataset/nuscenes",
                "lyft_sample": "../behavior-generation-dataset/lyft_prediction/scenes/sample.zarr",
                "lyft_val": "../behavior-generation-dataset/lyft_prediction/scenes/validate.zarr",
                "lyft_train": "../behavior-generation-dataset/lyft_prediction/scenes/train.zarr",
                "nuplan_mini": "../behavior-generation-dataset/nuplan/dataset/nuplan-v1.1",
            },
            cache_location="~/.unified_data_cache",
            num_workers=os.cpu_count()//2,
            rebuild_cache=False,
            rebuild_maps=False,
            standardize_data=True,
            max_agent_num=max_agent_num, 
            max_neighbor_num=max_neighbor_num, 
        )
        print(f"# Data Samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=dataloader_batch_size,
            shuffle=True,
            collate_fn=dataset.get_collate_fn(),
            num_workers=6,
        )
    # elif dataset_loader_to_use == 'l5kit':
    #     assert dataset_to_use in ["lyft_val", "lyft_train"]
    #     from tbsim.datasets.l5kit_datamodules import L5MixedDataModule
    #     from tbsim.configs.registry import get_registered_experiment_config
    #     from tbsim.utils.config_utils import get_experiment_config_from_file, translate_l5kit_cfg
        
    #     config_name = 'l5_bc'
    #     # config_name = None
    #     config_file = '../behavior-generation/diffuser_trained_models/test/run144_lyft/config.json'
        
    #     if config_name is not None:
    #         cfg = get_registered_experiment_config(config_name)
    #         cfg.train.dataset_path = '../behavior-generation-dataset/lyft_prediction'
    #     elif config_file is not None:
    #         # Update default config with external json file
    #         cfg = get_experiment_config_from_file(config_file, locked=False)


    #     cfg.lock()  # Make config read-only
    #     if not cfg.devices.num_gpus > 1:
    #         # Override strategy when training on a single GPU
    #         with cfg.train.unlocked():
    #             cfg.train.parallel_strategy = None

    #     l5_config = translate_l5kit_cfg(cfg)

    #     datamodule = L5MixedDataModule(l5_config=l5_config, train_config=cfg.train)
    #     datamodule.setup()
    #     if dataset_to_use == "lyft_val":
    #         dataloader = datamodule.val_dataloader()
    #     elif dataset_to_use == "lyft_train":
    #         dataloader = datamodule.train_dataloader()
    else:
        raise 


    batch: Union[AgentBatch, SceneBatch]
    compile_data = {
        key : [] for key in keys_to_compute
    }
    for i, batch in enumerate(tqdm(dataloader)):
        # print(batch.scene_ids)
        # print(batch.maps.size())
        # plot_agent_batch(batch, batch_idx=0, rgb_idx_groups=([1], [0], [1]))

        # normalize over future traj
        past_traj: Tensor = batch.agent_hist.cuda()
        future_traj: Tensor = batch.agent_fut.cuda()

        hist_pos, hist_yaw, hist_speed, _ = trajdata2posyawspeed(past_traj, nan_to_zero=False)
        curr_speed = hist_speed[..., -1]

        fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(future_traj, nan_to_zero=False)

        traj_state = torch.cat(
                (fut_pos, fut_yaw), dim=-1)
        traj_state_and_action = convert_state_to_state_and_action(traj_state, curr_speed, dt).reshape((-1, 6))

        traj_hist_state = torch.cat(
                (hist_pos[...,1:,:], hist_yaw[...,1:,:]), dim=-1)
        traj_hist_state_and_action = convert_state_to_state_and_action(traj_hist_state, hist_speed[...,-1], dt).reshape((-1, 6))

        if 'ego_fut' in compile_data:
            # B*T x 6 where (x, y, vel, yaw, acc, yawvel)
            compile_data['ego_fut'].append(traj_state_and_action.cpu().numpy())

        if 'ego_hist_diff' in compile_data:
            # B*T x 6 where (x, y, vel, yaw, acc, yawvel)
            compile_data['ego_hist_diff'].append(traj_hist_state_and_action.cpu().numpy())

        if 'ego_hist' in compile_data:
            # ego history (x, y, vel, l, w)
            ego_lw = batch.agent_hist_extent[...,:2].cuda()
            ego_hist_state = torch.cat((hist_pos, hist_speed.unsqueeze(-1), ego_lw), dim=-1).reshape((-1, 5))
            compile_data['ego_hist'].append(ego_hist_state.cpu().numpy())

        if 'neighbor_hist' in compile_data:
            # neighbor history
            neigh_hist_pos, _, neigh_hist_speed, neigh_mask = trajdata2posyawspeed(batch.neigh_hist.cuda(), nan_to_zero=False)
            neigh_lw = batch.neigh_hist_extents[...,:2].cuda()
            neigh_state = torch.cat((neigh_hist_pos, neigh_hist_speed.unsqueeze(-1), neigh_lw), dim=-1)
            # only want steps from neighbors that are valid
            neigh_state = neigh_state[neigh_mask]
            compile_data['neighbor_hist'].append(neigh_state.cpu().numpy())

        if 'neighbor_fut' in compile_data:
            # neighbor future
            neigh_fut_pos, _, neigh_fut_speed, neigh_mask = trajdata2posyawspeed(batch.neigh_fut.cuda(), nan_to_zero=False)
            neigh_lw = batch.neigh_fut_extents[...,:2].cuda()
            neigh_state = torch.cat((neigh_fut_pos, neigh_fut_speed.unsqueeze(-1), neigh_lw), dim=-1)
            # only want steps from neighbors that are valid
            neigh_state = neigh_state[neigh_mask]
            compile_data['neighbor_fut'].append(neigh_state.cpu().numpy())

        if steps is not None and i > steps:
            break

    
    compile_data = {state_name:np.concatenate(state_list, axis=0) for state_name, state_list in compile_data.items()}
    path = 'examples/traj_data_'+dataset_to_use+'_'+centric+'_'+str(hist_sec)+'_'+str(fut_sec)
    np.savez(path, **compile_data)
    print('data saved at', path)

def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
    else:
        assert state.shape[-1] == 8
    pos = state[..., :2]
    yaw = angle_wrap(torch.atan2(state[..., [-2]], state[..., [-1]]))
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

#
# Copied from our diffuser implementation so it's consistent
#
def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (..., 1)
    :param theta2: angle 2 (..., 1)
    :return diff: smallest angle difference between angles (..., 1)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

# TODO NEED TO HANLE MISSING FRAMES
def convert_state_to_state_and_action(traj_state, vel_init, dt, data_type='torch'):
    '''
    Infer vel and action (acc, yawvel) from state (x, y, yaw) based on Unicycle.
    Note:
        Support both agent-centric and scene-centric (extra dimension for the inputs).
    Input:
        traj_state: (batch_size, [num_agents], num_steps, 3)
        vel_init: (batch_size, [num_agents],)
        dt: float
        data_type: ['torch', 'numpy']
    Output:
        traj_state_and_action: (batch_size, [num_agents], num_steps, 6)
    '''
    BM = traj_state.shape[:-2]
    if data_type == 'torch':
        sin = torch.sin
        cos = torch.cos

        device = traj_state.get_device()
        pos_init = torch.zeros(*BM, 1, 2, device=device)
        yaw_init = torch.zeros(*BM, 1, 1, device=device)
    elif data_type == 'numpy':
        sin = np.sin
        cos = np.cos

        pos_init = np.zeros((*BM, 1, 2))
        yaw_init = np.zeros((*BM, 1, 1))
    else:
        raise
    def cat(arr, dim):
        if data_type == 'torch':
            return torch.cat(arr, dim=dim)
        elif data_type == 'numpy':
            return np.concatenate(arr, axis=dim)

    target_pos = traj_state[..., :2]
    traj_yaw = traj_state[..., 2:]    

    # pre-pad with zero pos and yaw
    pos = cat((pos_init, target_pos), dim=-2)
    yaw = cat((yaw_init, traj_yaw), dim=-2)

    # estimate speed from position and orientation
    vel_init = vel_init[..., None, None]
    vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * cos(
        yaw[..., 1:, :]
    ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * sin(
        yaw[..., 1:, :]
    )
    vel = cat((vel_init, vel), dim=-2)
    
    # m/s^2
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    # rad/s
    # yawvel = (yaw[..., 1:, :] - yaw[..., :-1, :]) / dt
    yawdiff = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :])
    yawvel = yawdiff / dt
    
    pos, yaw, vel = pos[..., 1:, :], yaw[..., 1:, :], vel[..., 1:, :]

    traj_state_and_action = cat((pos, vel, yaw, acc, yawvel), dim=-1)

    return traj_state_and_action

def compute_info(path, sample_coeff=0.25):
    compile_data_npz = np.load(path)

    val_labels = {
        'ego_fut' : [    'x',       ' y',       'vel',      'yaw',     'acc',    'yawvel' ],
        'ego_hist_diff' : [    'x',       ' y',       'vel',      'yaw',     'acc',    'yawvel' ],
        'ego_hist' : [    'x',        'y',       'vel',      'len',     'width'    ],
        'neighbor_hist' : [    'x',        'y',       'vel',      'len',     'width'    ],
        'neighbor_fut': [    'x',        'y',       'vel',      'len',     'width'    ],
    }
    for i, state_name in enumerate(compile_data_npz.files):
        print(state_name)
        all_states = compile_data_npz[state_name]
        all_states = all_states[:int(all_states.shape[0]*sample_coeff)]
        
        # all_states = np.concatenate(state_list, axis=0)
        print(all_states.shape)
        print(np.sum(np.isnan(all_states)))

        # import matplotlib
        # import matplotlib.pyplot as plt
        # for di, dname in enumerate(['x', 'y', 'vel', 'yaw', 'acc', 'yawvel']):
        #     fig = plt.figure()
        #     plt.hist((all_state_and_action[:,di] - np_mean[di]) / np_std[di], bins=100)
        #     plt.title(dname)
        #     plt.show()
        #     plt.close(fig)

        # remove outliers before computing final statistics
        print('Removing outliers...')
        med = np.nanmedian(all_states, axis=0, keepdims=True)
        d = np.abs(all_states - med)
        mdev = np.nanstd(all_states, axis=0, keepdims=True)
        s = d / mdev
        dev_thresh = 3.0
        chosen = s > dev_thresh
        all_states[chosen] = np.nan # reject outide of N deviations from median
        print('after outlier removal:')
        print(np.sum(chosen))
        print(np.sum(chosen, axis=0))
        print(np.sum(chosen) / (s.shape[0]*s.shape[1])) # removal rate

        out_mean = np.nanmean(all_states, axis=0)
        out_std = np.nanstd(all_states, axis=0)
        out_max = np.nanmax(all_states, axis=0)
        out_min = np.nanmin(all_states, axis=0)

        if state_name in val_labels:
            print('    '.join(val_labels[state_name]))
            out_fmt = ['( '] + ['%05f, ' for _ in val_labels[state_name]] + [' )']
            out_fmt = ''.join(out_fmt)
            print('out-mean')
            print(out_fmt % tuple(out_mean.tolist()))
            print('out-std')
            print(out_fmt % tuple(out_std.tolist()))
            print('out-max')
            print(out_fmt % tuple(out_max.tolist()))
            print('out-min')
            print(out_fmt % tuple(out_min.tolist()))

if __name__ == "__main__":
    # 'nusc_trainval', 'lyft_train', 'lyft_sample', 'nuplan_mini'
    dataset_to_use = 'nusc_trainval' # 'nuplan_mini' # 'nusc_trainval' # 'lyft_train'
    # 'unified', 'l5kit'
    dataset_loader_to_use = 'unified'
    # "scene", "agent"
    centric = "agent"
    # subset of ['ego_fut', 'ego_hist_diff', 'ego_hist', 'neighbor_hist', 'neighbor_fut']
    keys_to_compute = ['ego_fut', 'ego_hist_diff', 'ego_hist', 'neighbor_hist', 'neighbor_fut']
    hist_sec = 3.0 # 1.0, 3.0, 3.0
    fut_sec = 5.2 # 2.0, 5.2, 14.0
    steps = 10000
    agent_types = [AgentType.VEHICLE] # [AgentType.PEDESTRIAN] # [AgentType.VEHICLE]
    
    main(dataset_to_use, dataset_loader_to_use, centric, keys_to_compute, hist_sec, fut_sec, steps=steps, agent_types=agent_types)
    
    # path = 'examples/traj_data_nusc_trainval_agent_3.0_5.2.npz'
    # compute_info(path, sample_coeff=1.0)