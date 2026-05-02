#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import numpy as np
from skimage.draw import polygon
from pytorch_lightning.metrics.metric import Metric
from ..occ_head_plugin import calculate_birds_eye_view_parameters, gen_dx_bx
import copy
from shapely.geometry import Polygon

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded

def box3d_to_corners(box3d):
    device = box3d.device if isinstance(box3d, torch.Tensor) else None
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    
    if device is not None:
        corners = torch.tensor(corners, device=device, dtype=torch.float32)
    return corners

def check_collision(ego_box, boxes):
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
        boxes: tensor with shape [N, 7]
    '''
    if  boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly =  Polygon([(point[0], point[1]) for point in corners_box[i]])
        collision = ego_poly.intersects(box_poly)
        if collision:
            return True

    return False

def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0]+1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]


class PlanningMetric(Metric):
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future
        # self.reset()

        # Modified states to track binary collision indicators rather than per-timestep counts
        self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("L2", default=torch.zeros(self.n_future),dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        for t in range(n_future):
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision



    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, 'only supprt bs=1'
        for i in range(B):
            gt_trajs_copy = copy.deepcopy(gt_trajs[i])
            gt_trajs_copy = torch.cat([torch.zeros(1, 2, device=gt_trajs.device), gt_trajs_copy], dim=0)
            gt_box_coll = self.evaluate_single_coll(gt_trajs_copy[i], fut_boxes)
            if gt_box_coll[0]==1:
                return torch.zeros_like(gt_box_coll[1:]).to(self.obj_box_col.device),torch.zeros_like(gt_box_coll[1:]).to(self.obj_box_col.device),True
            gt_box_coll = gt_box_coll[1:]
            fut_boxes = fut_boxes[1:]
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes)
            
            coll_time_idx = torch.nonzero(box_coll, as_tuple=False)
            if coll_time_idx.numel() > 0:
                first_box_coll = coll_time_idx[0].item()
                obj_box_coll_sum[first_box_coll:] += 1

        return obj_coll_sum.to(self.obj_box_col.device), obj_box_coll_sum.to(self.obj_box_col.device),False

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        Compute L2 distance between predicted and ground truth trajectories
        
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def update(self, trajs, gt_trajs, gt_trajs_mask,  fut_boxes=None, scene_token=None):
        '''
        Update metrics with new batch
        
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        segmentation: torch.Tensor (B, n_future, 200, 200) or None
        fut_boxes: list of future boxes or None
        '''
        # 如果gt_trajs_mask 有非0值则跳过此次更新:
        if not gt_trajs_mask.all(): ## for incomplete gt, we do not count this sample
            print("Incomplete gt, skip this sample")
            return
        
        assert trajs.shape == gt_trajs.shape
        # print('nfuture:',trajs.shape)
        trajs[..., 0] = - trajs[..., 0]
        gt_trajs[..., 0] = - gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_binary, obj_box_coll_binary,jump = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], fut_boxes)
        if not jump:
            
            self.total += len(trajs)
        # Update binary collision counters
        self.obj_col += obj_coll_binary
        self.obj_box_col += obj_box_coll_binary
        self.L2 += L2.sum(dim=0)

    def compute(self):
        '''
        Return collision rate (binary indicator of any collision) and L2 metrics
        
        Implements CR(t) = (∑ᵢ₌₀ᴺ Iᵢ) > 0, N = t/0.5 formula
        where collision rate is defined as percentage of trajectories that have 
        at least one collision point.
        '''
        # 从临时文件读取场景数，如果不存在则使用默认值 72
        
        # with open('/tmp/nuscenes_scene_count.txt', 'r') as f:
        #     num_scenes = int(f.read().strip())
            # print('Read num_scenes from file:', num_scenes)
        num_scenes = 41
        # print('\ntotal_length:',self.total)
        # print("\nself.obj_box_col:",self.obj_box_col)
        return {
            # Return collision rate as percentage of trajectories with any collision
            # 'obj_col': self.obj_col / self.total,
            # 'obj_box_col': self.obj_box_col / self.total,
            # 'L2': self.L2 / self.total
            'obj_col': self.obj_box_col / self.total, #实际上就是sample-level CR
            'obj_box_col': self.obj_box_col / num_scenes, #实际上就是scene-level CR
            'L2': self.L2 / num_scenes
        }