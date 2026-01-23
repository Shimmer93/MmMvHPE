import torch
import torch.nn as nn

from models.leir import RegressorRGBPC


class LEIRAggregator(nn.Module):
    def __init__(
        self,
        rgb_in_dim: int = 384,
        rgb_proj_dim: int = 2048,
        smpl_model_path: str = 'weights/smpl/SMPL_NEUTRAL.pkl',
        use_pc_feature_in_batch: bool = False,
    ):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_in_dim, rgb_proj_dim)
        self.rgb_norm = nn.LayerNorm(rgb_proj_dim)
        self.regressor = RegressorRGBPC(
            smpl_model_path=smpl_model_path,
            use_pc_feature_in_batch=use_pc_feature_in_batch,
        )

    def forward(self, features, **batch):
        features_rgb, _, _, _ = features
        if features_rgb is None:
            raise ValueError("LEIRAggregator requires RGB features.")

        if features_rgb.dim() == 4:
            rgb_feat = features_rgb.mean(dim=2)
        else:
            rgb_feat = features_rgb

        rgb_feat = self.rgb_norm(self.rgb_proj(rgb_feat))

        if 'input_lidar' not in batch:
            raise KeyError("LEIRAggregator requires 'input_lidar' in batch.")

        data = {
            'img_feature': rgb_feat,
            'point': batch['input_lidar'],
        }
        if 'pc_feature' in batch:
            data['pc_feature'] = batch['pc_feature']

        preds = self.regressor(data)
        return preds
