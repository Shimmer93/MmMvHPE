import torch
import torch.nn as nn
from einops import rearrange

from misc.pose_enc import pose_encoding_to_extri_intri
from .base_head import BaseHead


class RegressionKeypointHeadV5(BaseHead):
    def __init__(
        self,
        losses,
        emb_size=512,
        num_joints=24,
        num_camera_tokens=1,
        num_register_tokens=4,
        num_smpl_tokens=1,
        max_modalities=4,
        last_n_layers=-1,
        pose_encoding_type="absT_quaR_FoV",
    ):
        super().__init__(losses)
        self.emb_size = emb_size
        self.num_joints = num_joints
        self.num_camera_tokens = num_camera_tokens
        self.num_register_tokens = num_register_tokens
        self.num_smpl_tokens = num_smpl_tokens
        self.max_modalities = max_modalities
        self.last_n_layers = last_n_layers
        self.pose_encoding_type = pose_encoding_type

        self.keypoint_gate = nn.Parameter(torch.zeros(emb_size))

        self.projector = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 3),
        )
        self.mlp_2d = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 2),
        )

        global_in = emb_size * max_modalities * 2
        self.global_projector = nn.Sequential(
            nn.LayerNorm(global_in),
            nn.Linear(global_in, emb_size),
            nn.ReLU(),
        )
        self.global_norm = nn.LayerNorm(emb_size)
        self.global_mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 3),
        )

    def forward(self, x, modalities=None):
        x = self._select_layers(x)
        if x.dim() == 4:
            x = x.unsqueeze(2)

        # gate = torch.sigmoid(self.keypoint_gate).view(1, 1, 1, 1, -1)
        # x = x * gate

        joint_tokens = self._extract_joint_tokens(x)
        camera_tokens = self._extract_camera_tokens(x)

        pred_per_modality = self._regress_per_modality(joint_tokens, modalities)
        pred_global = self._regress_global(joint_tokens, camera_tokens)

        return {"per_modality": pred_per_modality, "global": pred_global}

    def loss(self, x, data_batch):
        modalities = data_batch.get("modalities", [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]

        outputs = self.forward(x, modalities=modalities)
        pred_per_modality = outputs["per_modality"]
        pred_global = outputs["global"]

        gt_keypoints = data_batch.get("gt_keypoints", None)
        if gt_keypoints is None:
            return {}

        if isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = gt_keypoints.float()
        else:
            gt_keypoints = torch.as_tensor(gt_keypoints, dtype=torch.float32)
        if gt_keypoints.dim() == 2:
            gt_keypoints = gt_keypoints.unsqueeze(0)
        gt_keypoints = gt_keypoints.to(pred_global.device)

        losses = {}
        num_modalities = min(len(pred_per_modality), len(modalities))
        for i in range(num_modalities):
            modality = modalities[i]
            pred = pred_per_modality[i]
            proj_pred, proj_gt = self._project_keypoints(
                pred, gt_keypoints, modality, data_batch
            )
            if proj_pred is None:
                continue
            for loss_name, (loss_fn, loss_weight) in self.losses.items():
                losses[f"{loss_name}_{modality}"] = (loss_fn(proj_pred, proj_gt), loss_weight)

        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[f"{loss_name}_global"] = (loss_fn(pred_global, gt_keypoints), loss_weight)

        return losses

    def predict(self, x):
        outputs = self.forward(x)
        return outputs["global"]

    def _select_layers(self, x):
        if not isinstance(x, list):
            return x
        if self.last_n_layers > 0:
            x = x[-self.last_n_layers :]
        x = [xi[..., :xi.shape[-1]//2] for xi in x]
        return torch.cat(x, dim=-1)

    def _extract_joint_tokens(self, x):
        num_special = self.num_camera_tokens + self.num_register_tokens + self.num_smpl_tokens
        start = num_special
        end = start + self.num_joints
        joint_tokens = x[..., start:end, :]
        return joint_tokens.mean(dim=1)

    def _extract_camera_tokens(self, x):
        if self.num_camera_tokens <= 0:
            B, T, M, _, C = x.shape
            return torch.zeros(B, M, 1, C, device=x.device, dtype=x.dtype)
        camera_tokens = x[..., : self.num_camera_tokens, :]
        camera_tokens = camera_tokens.mean(dim=1)
        if camera_tokens.shape[2] > 1:
            camera_tokens = camera_tokens.mean(dim=2, keepdim=True)
        return camera_tokens

    def _regress_per_modality(self, joint_tokens, modalities=None):
        B, M, J, C = joint_tokens.shape
        preds = []
        if modalities is None:
            for i in range(M):
                feat_i = self.projector(joint_tokens[:, i])
                feat_i = self.norm(feat_i)
                pred = self.mlp(feat_i).reshape(B, J, 3)
                preds.append(pred)
            return preds

        for i, modality in enumerate(modalities[:M]):
            feat_i = self.projector(joint_tokens[:, i])
            feat_i = self.norm(feat_i)
            if modality.lower() in {"rgb", "depth"}:
                pred = self.mlp_2d(feat_i).reshape(B, J, 2)
            else:
                pred = self.mlp(feat_i).reshape(B, J, 3)
            preds.append(pred)
        return preds

    def _regress_global(self, joint_tokens, camera_tokens):
        B, M, J, C = joint_tokens.shape
        cam_expanded = camera_tokens.expand(-1, -1, J, -1)
        per_modality = torch.cat([joint_tokens, cam_expanded], dim=-1)

        if M < self.max_modalities:
            pad = torch.zeros(
                B, self.max_modalities - M, J, per_modality.shape[-1],
                device=per_modality.device, dtype=per_modality.dtype
            )
            per_modality = torch.cat([per_modality, pad], dim=1)
        elif M > self.max_modalities:
            per_modality = per_modality[:, : self.max_modalities, :, :]

        feats = rearrange(per_modality, "b m j c -> b j (m c)")
        feats = self.global_projector(feats)
        feats = self.global_norm(feats)
        return self.global_mlp(feats)

    def _project_keypoints(self, pred, gt, modality, data_batch):
        modality = modality.lower()
        if modality in {"rgb", "depth"}:
            gt_proj = self._get_2d_keypoints(gt, modality, data_batch, pred.device)
            if gt_proj is None:
                return None, None
            return pred, gt_proj
        if modality in {"lidar", "mmwave"}:
            gt_cam = self._get_pc_centered_keypoints(data_batch, modality, pred.device)
            if gt_cam is None:
                return None, None
            return pred, gt_cam

        return None, None

    def _get_2d_keypoints(self, gt, modality, data_batch, device):
        cam_params = self._get_camera_params(data_batch, modality, device)
        if cam_params is None:
            return None
        extrinsics, intrinsics, image_size = cam_params
        gt_proj = self._project_to_image(gt, extrinsics, intrinsics)
        gt_proj = self._normalize_2d(gt_proj, image_size)
        gt_proj = torch.clamp(gt_proj, -1.0, 1.0)
        return gt_proj

    def _get_pc_centered_keypoints(self, data_batch, modality, device):
        key = f"gt_keypoints_pc_centered_input_{modality}"
        gt = data_batch.get(key, None)
        if gt is None:
            return None
        if isinstance(gt, (list, tuple)):
            gt = [g for g in gt if g is not None]
            if len(gt) == 0:
                return None
            gt = gt[0]
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt, dtype=torch.float32)
        gt = gt.to(device).float()
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        return gt

    def _get_camera_params(self, data_batch, modality, device):
        gt_camera = data_batch.get(f"gt_camera_{modality}", None)
        if gt_camera is None:
            return None
        if not isinstance(gt_camera, torch.Tensor):
            gt_camera = torch.as_tensor(gt_camera, dtype=torch.float32)
        gt_camera = gt_camera.to(device).float()
        if gt_camera.dim() == 2:
            gt_camera = gt_camera.unsqueeze(0)
        if gt_camera.dim() == 3:
            gt_camera = gt_camera[:, -1]

        image_size = self._get_image_size(data_batch, modality)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            gt_camera.unsqueeze(1),
            image_size_hw=image_size,
            pose_encoding_type=self.pose_encoding_type,
            build_intrinsics=True,
        )
        return extrinsics.squeeze(1), intrinsics.squeeze(1), image_size

    def _get_image_size(self, data_batch, modality):
        input_key = f"input_{modality}"
        if input_key in data_batch:
            inp = data_batch[input_key]
            if isinstance(inp, torch.Tensor) and inp.dim() >= 4:
                return int(inp.shape[-2]), int(inp.shape[-1])
        return (224, 224)

    @staticmethod
    def _transform_to_camera(points, extrinsics):
        R = extrinsics[:, :3, :3]
        T = extrinsics[:, :3, 3]
        return torch.einsum("bij,bkj->bki", R, points) + T.unsqueeze(1)

    @staticmethod
    def _project_to_image(points, extrinsics, intrinsics):
        cam_points = RegressionKeypointHeadV5._transform_to_camera(points, extrinsics)
        cam_z = cam_points[..., 2].clamp(min=1e-6)
        proj = torch.einsum("bij,bkj->bki", intrinsics, cam_points)
        u = proj[..., 0] / cam_z
        v = proj[..., 1] / cam_z
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _normalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = points_2d[..., 0] / (width - 1) * 2.0 - 1.0
        y = points_2d[..., 1] / (height - 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)
