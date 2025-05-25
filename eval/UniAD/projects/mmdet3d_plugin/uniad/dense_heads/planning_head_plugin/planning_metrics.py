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


class PlanningMetric(Metric):
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)
        dx, bx, _ = gen_dx_bx([-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0])
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension = self.bev_dimension.numpy()

        self.W = 1.85  # ego width
        self.H = 4.084  # ego length
        self.imu_to_lidar_offset = 0.985793  # distance between IMU and LiDAR

        self.n_future = n_future

        # Modified states to track binary collision indicators rather than per-timestep counts
        self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("L2", default=torch.zeros(self.n_future),dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def evaluate_single_coll(self, traj, segmentation, input_gt=None, gt_traj=None, index=None):
        '''
        gt_segmentation
        traj: torch.Tensor (n_future, 2)
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs[:,:,[0,1]] = trajs[:,:,[1,0]] # can also change original tensor

        # trajs_ = copy.deepcopy(trajs)
        trajs = trajs / self.dx #.to(trajs.device)
        trajs= trajs.cpu().numpy() + rc # (n_future, 32, 2)

        r = trajs[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())
        return torch.from_numpy(collision).to(device=traj.device)
    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        '''
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])
            if gt_box_coll[0]==1:
                return torch.zeros_like(gt_box_coll),torch.zeros_like(gt_box_coll),True
            xx, yy = trajs[i,:,0], trajs[i,:,1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future, device=trajs.device)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()


            m2 = torch.logical_not(gt_box_coll)

            seg_mask = torch.ones_like(m2, dtype=torch.bool)
            for t in range(n_future):
                ratio = (segmentation[i, t]).float().mean()
                if ratio > 0.9:
                    seg_mask[t] = False
            m2 = seg_mask
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            
            effective_coll = torch.logical_and(box_coll, m2)
            coll_time_idx = torch.nonzero(effective_coll, as_tuple=False)
            if coll_time_idx.numel() > 0:
                first_box_coll = coll_time_idx[0].item()
                obj_box_coll_sum[first_box_coll:] += 1

        return obj_coll_sum, obj_box_coll_sum,False

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        Compute L2 distance between predicted and ground truth trajectories
        
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def update(self, trajs, gt_trajs, gt_trajs_mask, segmentation,scene_token=None):
        '''
        Update metrics with new batch
        
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        '''
        assert trajs.shape == gt_trajs.shape
        # print('nfuture:',trajs.shape)
        trajs[..., 0] = - trajs[..., 0]
        gt_trajs[..., 0] = - gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_binary, obj_box_coll_binary,jump = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation)
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

        return {
            'obj_col': self.obj_col / self.total,
            #sample-level collision rate
            'obj_box_col': self.obj_box_col / self.total, 
            # scene-level collision rate
            # 'obj_box_col': self.obj_box_col / {number of the scene in the datasets}  
            'L2': self.L2 / self.total
        }