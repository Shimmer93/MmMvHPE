import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.rotation import rot6d_to_rotmat
from .attention import PositionwiseFeedForward
from .pointnet2 import PointNet2Encoder
from .stgcn import STGCN
from .vibe import VIBERGBPC


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super().__init__()
        self.rnn = nn.GRU(
            n_hidden,
            n_hidden,
            n_rnn_layer,
            batch_first=True,
            bidirectional=True,
        )
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, x):
        x = self.rnn(F.relu(F.dropout(self.linear1(x)), inplace=True))[0]
        return self.linear2(x)


class RegressorRGBPC(nn.Module):
    def __init__(self, smpl_model_path, use_pc_feature_in_batch=False):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBERGBPC(
            smpl_model_path=smpl_model_path,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            bidirectional=False,
            use_residual=True,
        )
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
        self.use_pc_feature_in_batch = use_pc_feature_in_batch

    def forward(self, data):
        pred = {}
        if self.use_pc_feature_in_batch:
            pc_feature = data["pc_feature"]
        else:
            pc_feature = self.encoder(data["point"])

        B, T, D = pc_feature.shape

        img_feature, smpl_output = self.vibe(data["img_feature"], pc_feature=pc_feature.detach())
        img_feature_detached = self.fc(img_feature).view(B, T, D).detach()

        pc_res = pc_feature
        img_res = img_feature_detached
        img_feature = img_feature_detached

        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature, _ = self.self_attention(pc_feature, pc_feature, pc_feature)

        img_feature = self.trans_encoder1(img_feature)
        img_feature, _ = self.self_attention(img_feature, img_feature, img_feature)

        feature, _ = self.cross_attention(pc_feature, img_feature, img_feature)
        feature = self.ffn(feature)

        for _ in range(5):
            pc_feature, _ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature, _ = self.self_attention(img_feature, img_feature, img_feature)
            feature, _ = self.cross_attention(pc_feature, feature, img_feature)
            feature = self.ffn(feature)

        feature = feature + pc_res
        feature = self.layer_norm(feature)
        res_fusion = feature
        feature = self.ffn(feature)
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        feature = feature + pc_res + img_res

        joint = self.rnn(feature)
        stgcn_input = torch.cat(
            (
                joint.reshape(B, T, 24, 3),
                feature.unsqueeze(-2).repeat(1, 1, 24, 1),
            ),
            dim=-1,
        )
        with torch.cuda.amp.autocast(enabled=False):
            stgcn_input = stgcn_input.float()
            rot6ds = self.stgcn(stgcn_input)
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rot6ds = torch.nan_to_num(rot6ds, nan=0.0, posinf=0.0, neginf=0.0)
        rot6ds = torch.clamp(rot6ds, min=-5.0, max=5.0)
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)

        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
        return pred
