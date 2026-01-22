import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.rotation import rot6d_to_rotmat, rotation_matrix_to_angle_axis
from .attention import PositionwiseFeedForward, SelfAttention
from .smpl import LEIRSMPL


class TemporalEncoder(nn.Module):
    def __init__(self, n_layers=2, hidden_size=1024, add_linear=True, bidirectional=False, use_residual=True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers,
        )
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1, 0, 2)
        return y


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]


def projection(pred_joints, pred_camera):
    cam_scale = torch.clamp(pred_camera[:, 0], min=1e-2, max=10.0)
    pred_cam_t = torch.stack([
        pred_camera[:, 1],
        pred_camera[:, 2],
        2 * 5000.0 / (224.0 * cam_scale + 1e-9),
    ], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2, device=pred_joints.device)
    pred_keypoints_2d = perspective_projection(
        pred_joints,
        rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
        translation=pred_cam_t,
        focal_length=5000.0,
        camera_center=camera_center,
    )
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


class Regressor(nn.Module):
    def __init__(self, smpl_model_path):
        super().__init__()
        npose = 24 * 6
        self.smpl = LEIRSMPL(smpl_model_path)
        self.fc1 = nn.Linear(512 * 2 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.register_buffer('init_pose', torch.zeros(1, npose))
        self.register_buffer('init_shape', torch.zeros(1, 10))
        self.register_buffer('init_cam', torch.zeros(1, 3))

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for _ in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_pose = torch.nan_to_num(pred_pose, nan=0.0, posinf=0.0, neginf=0.0)
        pred_shape = torch.nan_to_num(pred_shape, nan=0.0, posinf=0.0, neginf=0.0)
        pred_cam = torch.nan_to_num(pred_cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam_scale = torch.clamp(pred_cam[:, 0], min=1e-2, max=100.0)
        pred_cam = torch.cat([cam_scale.unsqueeze(1), pred_cam[:, 1:]], dim=1)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_vertices = self.smpl(pred_rotmat, torch.zeros((batch_size, 10), device=x.device))
        pred_joints = self.smpl.get_full_joints(pred_vertices).reshape(batch_size, 24, 3)
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = {
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'rotmat': pred_rotmat,
        }
        return output


class VIBERGBPC(nn.Module):
    def __init__(self, smpl_model_path, n_layers=2, hidden_size=1024, add_linear=True, bidirectional=False, use_residual=True):
        super().__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor(smpl_model_path)
        self.cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, dropout=0.1)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, dropout=0.1)
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)
        self.trans_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=4),
            num_layers=6)
        self.trans_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=4),
            num_layers=12)

    def forward(self, input, pc_feature=None):
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        feature = feature.reshape(-1, feature.size(-1))
        img_feature = self.fc(feature)
        img_feature = img_feature.reshape(batch_size, seqlen, 1024)
        pc_res = pc_feature
        img_res = img_feature

        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature, _ = self.self_attention(pc_feature, pc_feature, pc_feature)

        img_feature = self.trans_encoder1(img_feature)
        img_feature, _ = self.self_attention(img_feature, img_feature, img_feature)

        fusion_feature, _ = self.cross_attention(pc_feature, img_feature, img_feature)
        fusion_feature = self.ffn(fusion_feature)

        for _ in range(5):
            pc_feature, _ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature, _ = self.self_attention(img_feature, img_feature, img_feature)
            fusion_feature, _ = self.cross_attention(pc_feature, fusion_feature, img_feature)
            fusion_feature = self.ffn(fusion_feature)

        fusion_feature = fusion_feature + pc_res
        fusion_feature = self.layer_norm(fusion_feature)
        res_fusion = fusion_feature
        fusion_feature = self.ffn(fusion_feature)
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        fusion_feature = fusion_feature + pc_res + img_res
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1))

        smpl_output = self.regressor(fusion_feature)
        return feature, smpl_output
