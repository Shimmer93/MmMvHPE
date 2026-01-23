import torch
import torch.nn as nn
import numpy as np

from models.smpl import SMPL
from misc.rotation import rot6d_to_rotmat, rotation_matrix_to_angle_axis, batch_rodrigues
from .base_head import BaseHead

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    
    # This function computes the perspective projection of a set of points.
    # Input:
    #     points (bs, N, 3): 3D points
    #     rotation (bs, 3, 3): Camera rotation
    #     translation (bs, 3): Camera translation
    #     focal_length (bs,) or scalar: Focal length
    #     camera_center (bs, 2): Camera center
    
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center
    
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    projected_points = points / points[:,:,-1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


class Regressor(nn.Module):
    def __init__(self, smpl_path, smpl_mean_params, emb_size=1024):
        super(Regressor, self).__init__()

        npose = 24 * 6
        self.smpl = SMPL(model_path=smpl_path)

        self.joint_fc1 = nn.Linear(emb_size + 6, emb_size)
        self.joint_fc2 = nn.Linear(emb_size, emb_size)
        self.decpose = nn.Linear(emb_size, 6)
        self.shape_fc1 = nn.Linear(emb_size + 10, emb_size)
        self.shape_fc2 = nn.Linear(emb_size, emb_size)
        self.decshape = nn.Linear(emb_size, 10)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):

        # x: (B*T, num_smpl_tokens + num_joints, C) # num_smpl_tokens=1, num_joints=24
        device = x.device
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B*T, N, C), got {tuple(x.shape)}.")
        batch_size, num_tokens, _ = x.shape
        num_smpl_tokens = 1
        num_joints = 24
        if num_tokens != num_smpl_tokens + num_joints:
            raise ValueError(
                f"Expected {num_smpl_tokens + num_joints} tokens, got {num_tokens}."
            )

        smpl_tokens = x[:, :num_smpl_tokens, :]
        joint_tokens = x[:, num_smpl_tokens:, :]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)

        pred_pose = init_pose.view(batch_size, num_joints, 6)
        pred_shape = init_shape

        for i in range(n_iter):
            joint_x = torch.cat([joint_tokens, pred_pose], dim=-1)
            joint_x = self.joint_fc1(joint_x)
            joint_x = self.drop1(joint_x)
            joint_x = self.joint_fc2(joint_x)
            joint_x = self.drop2(joint_x)

            smpl_x = torch.cat([smpl_tokens, pred_shape.unsqueeze(1)], dim=-1)
            smpl_x = self.shape_fc1(smpl_x)
            smpl_x = self.drop1(smpl_x)
            smpl_x = self.shape_fc2(smpl_x)
            smpl_x = self.drop2(smpl_x)

            pred_pose = self.decpose(joint_x) + pred_pose
            pred_shape = self.decshape(smpl_x.squeeze(1)) + pred_shape

        pred_pose_flat = pred_pose.reshape(batch_size, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose_flat).view(batch_size, 24, 3, 3)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        pose[:, :3] = 0.0  # Fix global rotation to zero
        
        smpl_model = self.smpl.to(device)
        output = smpl_model(
            pose,
            pred_shape,
            torch.zeros((batch_size, 3)).to(device),
        )
        pred_joints = output[1] if isinstance(output, (list, tuple)) else output

        output = {'pred_smpl_params'  : torch.cat([pose, pred_shape], dim=1), # (B*T, 82) 82 = 72 + 10
                  'pred_keypoints'  : pred_joints, # (B*T, 24, 3)
                  'pred_rotmat' : pred_rotmat } # (B*T, 24, 3, 3)
        return output

class VIBETokenHead(BaseHead):
    def __init__(self, losses, smpl_path, smpl_mean_params, emb_size, n_iters=3, last_n_layers=-1):
        super().__init__(losses)
        self.emb_size = emb_size

        self.projector = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU()
        )

        self.regressor = Regressor(smpl_path, smpl_mean_params, emb_size=emb_size)
        self.n_iters = n_iters
        self.last_n_layers = last_n_layers

    def forward(self, x):
        if isinstance(x, list):
            if self.last_n_layers > 0:
                x = x[-self.last_n_layers:]
            x = [x_[..., :x_.shape[-1] // 2] for x_ in x]
            x = torch.concatenate(x, dim=-1)
        # x.shape: B, T, num_camera_tokens + num_smpl_tokens + num_joints, C
        B, T, N, C = x.shape
        num_camera_tokens = N - 1 - 24
        x = x[:, -1, num_camera_tokens:, :]  # (B, num_smpl_tokens + num_joints, C)

        x = self.projector(x)
        output = self.regressor(x, n_iter=self.n_iters)
        
        output = {
            'pred_smpl_params': output['pred_smpl_params'],
            'pred_keypoints': output['pred_keypoints'],
            'pred_rotmat': output['pred_rotmat'],
        }
        return output
    
    def loss(self, x, data_batch):
        pred_output = self.forward(x)
        losses = {}
        gt_rotmat = data_batch.get('gt_rotmat', None)
        if gt_rotmat is None:
            gt_pose = data_batch['gt_smpl_params'][:, :72].contiguous()
            gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
            data_batch['gt_rotmat'] = gt_rotmat

        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            if 'keypoint' in loss_name.lower():
                losses[loss_name] = (loss_fn(pred_output['pred_keypoints'], data_batch['gt_keypoints']), loss_weight)
            elif 'smplpose' in loss_name.lower():
                losses[loss_name] = (loss_fn(pred_output['pred_smpl_params'][:, :72], data_batch['gt_smpl_params'][:, :72]), loss_weight)
            elif 'smplshape' in loss_name.lower():
                losses[loss_name] = (loss_fn(pred_output['pred_smpl_params'][:, 72:], data_batch['gt_smpl_params'][:, 72:]), loss_weight)
            elif 'rotmat' in loss_name.lower():
                losses[loss_name] = (loss_fn(pred_output['pred_rotmat'], data_batch['gt_rotmat']), loss_weight)

        return losses
    
    def predict(self, x):
        return self.forward(x)
        
        

    
