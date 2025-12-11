from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentType, UnifiedDataset
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg, parse_trajdata_batch, convert_scene_data_to_agent_coordinates, check_consistency, major_keywords, prep_keywords, neighbor_keywords

import numpy as np
import torch
from collections import defaultdict
from tbsim.utils.trajdata_utils import get_closest_lane_point
import matplotlib.pyplot as plt
from dataclasses import asdict

def main(dataset_to_use, hist_sec = 1.0, fut_sec = 2.0, agent_types = [AgentType.VEHICLE]):
    dt = 0.1

    max_neighbor_num = None
    max_neighbor_dist = np.inf

    raster_cfg = {
        "include_hist": True,
        "num_sem_layers": 3,
        "drivable_layers": None,
        "rgb_idx_groups": ([0], [1], [2]),
        "pixel_size": 1.0 / 2.0,
        "raster_size": 224,
        "ego_center": (-0.5, 0.0),
        "no_map_fill_value": -1.0,
    }
    
    set_global_trajdata_batch_env(dataset_to_use)
    set_global_trajdata_batch_raster_cfg(raster_cfg)
    
    dataset_scene = UnifiedDataset(
        desired_data=[dataset_to_use],
        centric='scene',
        desired_dt=dt,
        history_sec=(hist_sec, hist_sec),
        future_sec=(fut_sec, fut_sec), # This should be consistent with the horizon diffusion model uses
        only_types=agent_types,
        only_predict=agent_types,
        agent_interaction_distances=defaultdict(lambda: np.inf),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
        incl_vector_map=True,
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
        num_workers=0,
        rebuild_cache=False,
        rebuild_maps=False,
        standardize_data=True,
        # max_agent_num=max_neighbor_num+1,
        extras={
                "closest_lane_point": get_closest_lane_point,
        },
    )
    print(f"# Data Samples: {len(dataset_scene):,}")

    dataset_agent = UnifiedDataset(
        desired_data=[dataset_to_use],
        centric='agent',
        desired_dt=dt,
        history_sec=(hist_sec, hist_sec),
        future_sec=(fut_sec, fut_sec), # This should be consistent with the horizon diffusion model uses
        only_types=agent_types,
        only_predict=agent_types,
        agent_interaction_distances=defaultdict(lambda: max_neighbor_dist),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
        incl_vector_map=True,
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
        num_workers=0,
        rebuild_cache=False,
        rebuild_maps=False,
        standardize_data=True,
        max_neighbor_num=max_neighbor_num,
        extras={
                "closest_lane_point": get_closest_lane_point,
        },
    )
    print(f"# Data Samples: {len(dataset_agent):,}")

    dataloader_scene = DataLoader(
        dataset_scene,
        batch_size=2, # 1, 2
        shuffle=False,
        collate_fn=dataset_scene.get_collate_fn(return_dict=False),
        num_workers=0,
    )  

    dataloader_agent = DataLoader(
        dataset_agent,
        batch_size=12, # 10, 12
        shuffle=False,
        collate_fn=dataset_agent.get_collate_fn(return_dict=False),
        num_workers=0,
    )

    for i, (batch_scene, batch_agent) in enumerate(zip(dataloader_scene, dataloader_agent)):
        
        from tbsim.utils.trajdata_utils import plot_agent_batch_dict
        # from trajdata.visualization.vis import plot_agent_batch, plot_agent_batch_all, plot_scene_batch
        
        # ax = plot_agent_batch(batch_agent, batch_idx=3, legend=False, show=False, close=False)
        # plt.show()
        # raise

        batch_agent = asdict(batch_agent)
        batch_scene = asdict(batch_scene)

        batch_scene_parsed = parse_trajdata_batch(batch_scene, overwrite_nan=True)
        batch_scene_agent_coord = convert_scene_data_to_agent_coordinates(batch_scene_parsed, merge_BM=True, max_neighbor_num=max_neighbor_num, max_neighbor_dist=max_neighbor_dist)

        batch_agent_parsed = parse_trajdata_batch(batch_agent, overwrite_nan=True)
        
        
        # ax2 = plot_agent_batch_dict(batch_agent_parsed, batch_idx=5, legend=False, show=False, close=False)
        # plt.show()
        # raise

        
        # print('batch_scene.keys()', batch_scene.keys())
        # print('batch_scene["agent_names"]', batch_scene['agent_names'])
        
        # print('batch_scene_agent_coord.keys()', batch_scene_agent_coord.keys())
        # print('batch_agent_parsed.keys()', batch_agent_parsed.keys())
        # print('batch_agent_parsed["agent_name"]', batch_agent_parsed['agent_name'])

        # print("batch_scene_agent_coord['agent_names']", batch_scene_agent_coord['agent_names'])
        included_inds = []

        # TBD: hacky way to deal with ego (a better way is to consider scene name as well)
        k = 0
        for name in batch_agent_parsed['agent_name']:
            if name != 'ego':
                ind = np.where(batch_scene_agent_coord['agent_names']==name)[0][0]
            else:
                ind = np.where(batch_scene_agent_coord['agent_names']=='ego')[0][k]
                k += 1
            included_inds.append(ind)
        print('included_inds', included_inds)

        batch_scene_agent_coord_selected = {}
        for k, v in batch_scene_agent_coord.items():
            if k not in ['agent_names', 'extras']:
                batch_scene_agent_coord_selected[k] = v[included_inds]
            elif k == 'extras':
                batch_scene_agent_coord_selected[k] = {'closest_lane_point': v['closest_lane_point'][included_inds]}
            else:
                batch_scene_agent_coord_selected[k] = np.array(v)[included_inds]

        # ax3 = plot_agent_batch_dict(batch_scene_agent_coord_selected, batch_idx=5, legend=False, show=False, close=False)
        # plt.show()
        # raise
        
        # (B, M, T) -> (B, M) -> (B) -> ()
        max_len = torch.max(torch.sum(torch.sum(batch_scene_agent_coord_selected["all_other_agents_history_availabilities"], dim=-1)>0, dim=-1)).item()
        # print('max_len', max_len)
        for k in neighbor_keywords:
            # print(k, batch_scene_agent_coord_selected[k].shape)
            batch_scene_agent_coord_selected[k] = batch_scene_agent_coord_selected[k][:, :max_len]
            # print(k, batch_scene_agent_coord_selected[k].shape)

        check_consistency(prep_keywords, batch_scene_agent_coord_selected, batch_agent_parsed)
        check_consistency(major_keywords, batch_scene_agent_coord_selected, batch_agent_parsed)
        
        # print('batch_scene.num_agents', batch_scene['num_agents'])
        # print("batch_scene['scene_ts']", batch_scene['scene_ts'])
        # print('batch_scene.agent_fut.shape', batch_scene['agent_fut'].shape)

        # print("batch_agent['scene_ids']", batch_agent['scene_ids'])
        # print("batch_agent['scene_ts']", batch_agent['scene_ts'])
        # print('batch_agent.agent_fut.shape', batch_agent['agent_fut'].shape)
        # print('batch_agent.neigh_fut.shape', batch_agent['neigh_fut'].shape)
        raise
        

if __name__ == "__main__":
    # 'nusc_trainval', 'lyft_train', 'lyft_sample', 'nuplan_mini'
    dataset_to_use = 'nusc_trainval' # 'nuplan_mini' # 'nusc_trainval' # 'lyft_train'
    hist_sec = 3.0 # 1.0, 3.0, 3.0
    fut_sec = 16.5 # 2.0, 5.2, 14.0
    agent_types = [AgentType.VEHICLE] # [AgentType.PEDESTRIAN] # [AgentType.VEHICLE]
    
    main(dataset_to_use,hist_sec, fut_sec, agent_types=agent_types)
    