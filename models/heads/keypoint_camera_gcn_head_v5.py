import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .base_head import BaseHead
from misc.pose_enc import pose_encoding_to_extri_intri
from models.aggregators.layers.gcn import TCN_GCN_unit
from misc.skeleton import (
    get_adjacency_matrix,
    COCOSkeleton,
    SimpleCOCOSkeleton,
    MMBodySkeleton,
    H36MSkeleton,
    MiliPointSkeleton,
    SMPLSkeleton,
)


class KeypointCameraGCNHeadV5(BaseHead):
    """Predict camera pose encodings from keypoints using separate 2D/3D ST-GCNs."""

    def __init__(
        self,
        losses,
        num_joints: int = 24,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_modality_embedding: bool = True,
        use_gt_ratio: float = 0.0,
        modalities_key: str = "modalities",
        pose_encoding_dim: int = 9,
        pose_encoding_type: str = "absT_quaR_FoV",
        proj_loss_weight_rgb: float = 1.0,
        proj_loss_weight_lidar: float = 1.0,
        proj_loss_type: str = "l1",
        detach_inputs: bool = True,
        num_iterations: int = 1,
        gcn_kernel_size: int = 5,
        gcn_dilations: tuple = (1, 2),
        gcn_depth: int = 2,
        gcn_norm_type: str = "gn",
        gcn_num_groups: int = 32,
        input_2d_skeleton_format: str = "coco",
        input_3d_skeleton_format: str = "smpl",
        train_branch: str = "both",
    ):
        super().__init__(losses)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.use_modality_embedding = use_modality_embedding
        self.use_gt_ratio = float(use_gt_ratio)
        self.modalities_key = modalities_key
        self.pose_encoding_dim = pose_encoding_dim
        self.pose_encoding_type = pose_encoding_type
        self.proj_loss_weight_rgb = proj_loss_weight_rgb
        self.proj_loss_weight_lidar = proj_loss_weight_lidar
        self.proj_loss_type = proj_loss_type
        self.detach_inputs = detach_inputs
        self.num_iterations = int(max(1, num_iterations))
        self.gcn_depth = int(max(1, gcn_depth))
        self.gcn_norm_type = str(gcn_norm_type).lower()
        self.gcn_num_groups = int(max(1, gcn_num_groups))
        self.input_2d_skeleton_format = str(input_2d_skeleton_format).lower()
        self.input_3d_skeleton_format = str(input_3d_skeleton_format).lower()
        self.train_branch = str(train_branch).lower()
        if self.train_branch not in {"both", "2d", "3d"}:
            raise ValueError(
                f"Unsupported train_branch: {train_branch}. Expected one of ['both', '2d', '3d']."
            )

        self.target_skeleton = SimpleCOCOSkeleton()
        self.num_joints = self.target_skeleton.num_joints
        if num_joints != self.num_joints:
            warnings.warn(
                f"KeypointCameraGCNHeadV5 uses SimpleCOCO skeleton ({self.num_joints} joints). "
                f"Ignoring configured num_joints={num_joints}."
            )
        A = get_adjacency_matrix(self.target_skeleton.bones, self.target_skeleton.num_joints)

        self.gcn_in_channels = 9
        self.gcn_proj_2d = nn.Conv2d(5, self.gcn_in_channels, kernel_size=1)
        self.gcn_proj_3d = nn.Conv2d(6, self.gcn_in_channels, kernel_size=1)
        self.gcn_2d = self._build_gcn_stack(
            in_channels=self.gcn_in_channels,
            out_channels=hidden_dim,
            A=A,
            gcn_depth=self.gcn_depth,
            gcn_kernel_size=gcn_kernel_size,
            gcn_dilations=gcn_dilations,
            norm_type=self.gcn_norm_type,
            num_groups=self.gcn_num_groups,
        )
        self.gcn_3d = self._build_gcn_stack(
            in_channels=self.gcn_in_channels,
            out_channels=hidden_dim,
            A=A,
            gcn_depth=self.gcn_depth,
            gcn_kernel_size=gcn_kernel_size,
            gcn_dilations=gcn_dilations,
            norm_type=self.gcn_norm_type,
            num_groups=self.gcn_num_groups,
        )

        if use_modality_embedding:
            self.modality_embed = nn.Embedding(8, hidden_dim)
            mlp_in_dim = hidden_dim * 2
        else:
            self.modality_embed = None
            mlp_in_dim = hidden_dim

        self.prev_proj_2d = nn.Linear(pose_encoding_dim, mlp_in_dim)
        self.prev_proj_3d = nn.Linear(pose_encoding_dim, mlp_in_dim)
        self.mlp_2d = self._build_mlp(mlp_in_dim)
        self.mlp_3d = self._build_mlp(mlp_in_dim)
        self.post_gcn_norm_2d = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.post_gcn_norm_3d = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def forward(self, x, data_batch=None, pred_dict=None):
        return self.predict(x, data_batch=data_batch, pred_dict=pred_dict)

    def loss(self, x, data_batch, pred_dict=None):
        num_iterations = data_batch.get("num_camera_iterations", self.num_iterations)
        pred_encodings = self.predict(
            x, data_batch=data_batch, pred_dict=pred_dict, num_iterations=num_iterations
        )
        if pred_encodings is None:
            return {}

        modalities = self._get_modalities(data_batch)
        stream_modalities, stream_sensor_indices = self._build_stream_specs(
            modalities=modalities,
            data_batch=data_batch,
            target_streams=pred_encodings.shape[1],
        )
        if not stream_modalities:
            return {}

        losses = {}
        for m_idx, (modality, sensor_idx) in enumerate(zip(stream_modalities, stream_sensor_indices)):
            if not self._is_branch_enabled(modality):
                continue
            gt_camera = self._get_gt_camera_encoding(
                data_batch,
                modality.lower(),
                pred_encodings.device,
                pred_encodings.dtype,
                pred_encodings.shape[0],
                sensor_idx=sensor_idx,
            )
            if gt_camera is None:
                continue

            pred = pred_encodings[:, m_idx]
            valid = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt_camera).all(dim=-1)
            if valid.sum().item() == 0:
                continue
            pred = pred[valid]
            gt = gt_camera[valid]

            for loss_name, (loss_fn, loss_weight) in self.losses.items():
                loss_output = loss_fn(pred, gt)
                if isinstance(loss_output, dict):
                    for k, v in loss_output.items():
                        losses[f"{loss_name}_{modality}_s{sensor_idx}_{k}"] = (v, loss_weight)
                else:
                    losses[f"{loss_name}_{modality}_s{sensor_idx}"] = (loss_output, loss_weight)
        per_modality_preds = [pred_encodings[:, i, ...] for i in range(pred_encodings.shape[1])]
        stream_specs = list(zip(stream_modalities, stream_sensor_indices))
        proj_losses = self._projection_losses(per_modality_preds, stream_specs, data_batch)
        losses.update(proj_losses)
        return losses

    def predict(self, x, data_batch=None, pred_dict=None, num_iterations=None):
        if data_batch is None:
            return None

        modalities = self._get_modalities(data_batch)
        if not modalities:
            return None

        global_kps = self._get_global_keypoints(data_batch, pred_dict)
        if global_kps is None:
            return None

        global_kps = self._maybe_detach(global_kps)
        global_kps = self._convert_to_target_skeleton(
            global_kps,
            source_format=self.input_3d_skeleton_format,
        )
        if global_kps is None:
            return None
        device = global_kps.device
        dtype = global_kps.dtype
        batch_size = global_kps.shape[0] if global_kps.dim() >= 3 else 1
        global_kps = self._ensure_bstjc(global_kps, batch_size)
        if global_kps is None:
            return None
        if global_kps.shape[-1] != 3:
            raise ValueError(
                f"global_kps must have last dimension 3 (x, y, z), got shape={tuple(global_kps.shape)}"
            )
        num_streams_from_x = self._infer_stream_count_from_x(x)
        stream_modalities, stream_sensor_indices = self._build_stream_specs(
            modalities=modalities,
            data_batch=data_batch,
            target_streams=num_streams_from_x,
        )
        if not stream_modalities:
            return None

        pred_encodings = torch.full(
            (batch_size, len(stream_modalities), self.pose_encoding_dim),
            float("nan"),
            device=device,
            dtype=dtype,
        )

        if num_iterations is None:
            num_iterations = self.num_iterations
        num_iterations = int(max(1, num_iterations))

        for m_idx, (modality, sensor_idx) in enumerate(zip(stream_modalities, stream_sensor_indices)):
            modality_l = modality.lower()
            if modality_l in {"rgb", "depth"}:
                kps = self._get_keypoints_2d(
                    data_batch, pred_dict, modality_l, device, dtype, sensor_idx=sensor_idx
                )
            else:
                kps = self._get_keypoints_3d(
                    data_batch, pred_dict, modality_l, device, dtype, sensor_idx=sensor_idx
                )
            if kps is None:
                continue

            kps = self._maybe_detach(kps)
            kps = self._select_sensor_view(kps, sensor_idx=sensor_idx, batch_size=batch_size)
            kps = self._ensure_bstjc(kps, batch_size)
            if kps is None:
                continue
            if modality_l in {"rgb", "depth"}:
                kps = self._convert_to_target_skeleton(
                    kps,
                    source_format=self.input_2d_skeleton_format,
                )
            else:
                kps = self._convert_to_target_skeleton(
                    kps,
                    source_format=self.input_3d_skeleton_format,
                )
            if kps is None:
                continue
            global_seq, kps = self._align_temporal_length(global_kps, kps)
            if global_seq is None:
                continue
            if modality_l in {"rgb", "depth"}:
                global_3d = global_seq
                kps_2d = kps[..., :2] if kps.shape[-1] >= 2 else kps
                if kps_2d.shape[-1] != 2:
                    continue
                feat = torch.cat([global_3d, kps_2d], dim=-1)
                gcn_in = feat.permute(0, 3, 1, 2)
                gcn_in = self.gcn_proj_2d(gcn_in)
                gcn_out = self.gcn_2d(gcn_in)
                prev_proj = self.prev_proj_2d
                mlp = self.mlp_2d
                pooled = self.post_gcn_norm_2d(gcn_out.mean(dim=-1).mean(dim=-1))
            else:
                global_3d = global_seq
                kps_3d = kps
                if kps_3d.shape[-1] != 3:
                    continue
                feat = torch.cat([global_3d, kps_3d], dim=-1)
                gcn_in = feat.permute(0, 3, 1, 2)
                gcn_in = self.gcn_proj_3d(gcn_in)
                gcn_out = self.gcn_3d(gcn_in)
                prev_proj = self.prev_proj_3d
                mlp = self.mlp_3d
                pooled = self.post_gcn_norm_3d(gcn_out.mean(dim=-1).mean(dim=-1))
            if self.use_modality_embedding and self.modality_embed is not None:
                mod_id = torch.full(
                    (batch_size,),
                    self._modality_to_id(modality_l),
                    device=device,
                    dtype=torch.long,
                )
                pooled = torch.cat([pooled, self.modality_embed(mod_id)], dim=-1)

            pred = None
            for _ in range(num_iterations):
                if pred is not None:
                    pred = pred.detach()
                    feat_iter = pooled + prev_proj(pred)
                else:
                    feat_iter = pooled
                delta = mlp(feat_iter)
                pred = delta if pred is None else pred + delta

            pred_encodings[:, m_idx] = pred

        return pred_encodings

    def _build_mlp(self, input_dim):
        layers = []
        in_dim = input_dim
        for idx in range(max(1, self.num_layers)):
            out_dim = self.hidden_dim if idx < self.num_layers - 1 else self.pose_encoding_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < self.num_layers - 1:
                if self.use_layernorm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        return nn.Sequential(*layers)

    def _build_gcn_stack(
        self,
        in_channels,
        out_channels,
        A,
        gcn_depth,
        gcn_kernel_size,
        gcn_dilations,
        norm_type="gn",
        num_groups=32,
    ):
        layers = []
        for idx in range(gcn_depth):
            layers.append(
                TCN_GCN_unit(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    A=A,
                    residual=idx != 0,
                    kernel_size=gcn_kernel_size,
                    dilations=list(gcn_dilations),
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
        return nn.Sequential(*layers)


    def _get_modalities(self, data_batch):
        modalities = data_batch.get(self.modalities_key, [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        return modalities

    @staticmethod
    def _normalize_modalities(modalities):
        if modalities is None:
            return []
        if isinstance(modalities, (list, tuple)) and len(modalities) > 0 and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        if not isinstance(modalities, (list, tuple)):
            return []
        return [str(m).lower() for m in modalities]

    @staticmethod
    def _modality_to_id(modality: str) -> int:
        modality = str(modality).lower()
        if modality == "rgb":
            return 0
        if modality == "depth":
            return 1
        if modality == "lidar":
            return 2
        if modality == "mmwave":
            return 3
        return 0

    def _infer_stream_count_from_x(self, x):
        if isinstance(x, list):
            if len(x) == 0:
                return 0
            x = x[-1]
        if isinstance(x, torch.Tensor):
            if x.dim() >= 5:
                return int(x.shape[2])
            if x.dim() == 4:
                return int(x.shape[1])
        return 0

    def _build_stream_specs(self, modalities, data_batch, target_streams=None):
        modalities = self._normalize_modalities(modalities)
        specs = []
        for modality in modalities:
            n_sensors = self._infer_sensor_count(data_batch, modality)
            for sensor_idx in range(max(1, int(n_sensors))):
                specs.append((modality, sensor_idx))

        if target_streams is not None:
            target_streams = int(max(0, target_streams))
            if len(specs) > target_streams:
                specs = specs[:target_streams]
            elif len(specs) < target_streams:
                base_modalities = modalities if modalities else ["rgb"]
                counts = {}
                for m, s in specs:
                    counts[m] = max(counts.get(m, 0), s + 1)
                cursor = 0
                while len(specs) < target_streams:
                    m = base_modalities[cursor % len(base_modalities)]
                    s = counts.get(m, 0)
                    specs.append((m, s))
                    counts[m] = s + 1
                    cursor += 1

        return [m for m, _ in specs], [s for _, s in specs]

    def _infer_sensor_count(self, data_batch, modality):
        if not isinstance(data_batch, dict):
            return 1

        selected = data_batch.get("selected_cameras", None)
        if isinstance(selected, (list, tuple)) and len(selected) > 0:
            selected = selected[0]
        if isinstance(selected, dict):
            cams = selected.get(modality, None)
            if isinstance(cams, (list, tuple)) and len(cams) > 0:
                return len(cams)

        gt_camera = data_batch.get(f"gt_camera_{modality}", None)
        if isinstance(gt_camera, torch.Tensor):
            batch_size = self._infer_batch_size(data_batch)
            if gt_camera.dim() >= 4:
                return int(gt_camera.shape[1])
            if gt_camera.dim() == 3 and (batch_size is None or gt_camera.shape[0] != batch_size):
                return int(gt_camera.shape[0])
            if gt_camera.dim() == 2 and gt_camera.shape[1] == self.pose_encoding_dim and (
                batch_size is None or gt_camera.shape[0] != batch_size
            ):
                return int(gt_camera.shape[0])

        inp = data_batch.get(f"input_{modality}", None)
        if isinstance(inp, torch.Tensor):
            if modality in {"rgb", "depth"} and inp.dim() >= 6:
                return int(inp.shape[1])
            if modality in {"lidar", "mmwave"} and inp.dim() >= 5:
                return int(inp.shape[1])
        return 1

    @staticmethod
    def _infer_batch_size(data_batch):
        if not isinstance(data_batch, dict):
            return None
        gt_global = data_batch.get("gt_keypoints", None)
        if isinstance(gt_global, torch.Tensor):
            if gt_global.dim() >= 3:
                return int(gt_global.shape[0])
            return 1
        sample_ids = data_batch.get("sample_id", None)
        if isinstance(sample_ids, (list, tuple)):
            return len(sample_ids)
        return None

    @staticmethod
    def _select_sensor_view(tensor, sensor_idx, batch_size):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.dim() == 5:
            if tensor.shape[0] == batch_size:
                view_idx = min(sensor_idx, tensor.shape[1] - 1)
                return tensor[:, view_idx]
            if batch_size == 1:
                view_idx = min(sensor_idx, tensor.shape[0] - 1)
                return tensor[view_idx].unsqueeze(0)
            return None
        if tensor.dim() == 4:
            if tensor.shape[0] == batch_size:
                return tensor
            if batch_size == 1:
                view_idx = min(sensor_idx, tensor.shape[0] - 1)
                return tensor[view_idx : view_idx + 1]
            return None
        if tensor.dim() == 3:
            if tensor.shape[0] == batch_size:
                return tensor
            if batch_size == 1:
                view_idx = min(sensor_idx, tensor.shape[0] - 1)
                return tensor[view_idx : view_idx + 1]
            return None
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        return tensor

    def _get_global_keypoints(self, data_batch, pred_dict):
        pred = None
        if pred_dict is not None:
            pred = pred_dict.get("pred_keypoints")
        gt = data_batch.get("gt_keypoints")
        keypoints = self._mix_pred_gt(pred, gt)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints)
        return keypoints

    def _get_keypoints_2d(self, data_batch, pred_dict, modality, device, dtype, sensor_idx=0):
        pred = None
        if pred_dict is not None:
            pred = pred_dict.get(f"pred_keypoints_2d_{modality}_s{sensor_idx}")
            if pred is None:
                pred = pred_dict.get(f"pred_keypoints_2d_{modality}")
        gt = data_batch.get(f"gt_keypoints_2d_{modality}")
        if modality == "rgb":
            # Prefer dataset-provided RGB 2D skeletons (e.g., JSON-loaded),
            # and only fall back to predicted RGB keypoints if unavailable.
            keypoints = gt if gt is not None else pred
        else:
            keypoints = self._mix_pred_gt(pred, gt)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints).to(device=device, dtype=dtype)
        return keypoints

    def _get_keypoints_3d(self, data_batch, pred_dict, modality, device, dtype, sensor_idx=0):
        pred = None
        if pred_dict is not None:
            if modality == "lidar":
                pred = pred_dict.get(f"pred_keypoints_pc_centered_input_lidar_s{sensor_idx}")
                if pred is None:
                    pred = pred_dict.get("pred_keypoints_pc_centered_input_lidar")
            if pred is None:
                pred = pred_dict.get(f"pred_keypoints_3d_{modality}_s{sensor_idx}")
            if pred is None:
                pred = pred_dict.get(f"pred_keypoints_3d_{modality}")
        gt = data_batch.get(f"gt_keypoints_{modality}")
        if gt is None and modality == "lidar":
            gt = data_batch.get("gt_keypoints_pc_centered_input_lidar")
        keypoints = self._mix_pred_gt(pred, gt)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints).to(device=device, dtype=dtype)
        return keypoints

    @staticmethod
    def _to_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None

            items = []
            ref = None
            for item in x:
                t = KeypointCameraGCNHeadV5._to_tensor(item)
                if t is None:
                    items.append(None)
                    continue
                t = t.float()
                if ref is None:
                    ref = t
                items.append(t)

            if ref is None:
                return None

            ref_shape = ref.shape
            ref_device = ref.device
            ref_dtype = ref.dtype
            stacked = []
            for t in items:
                if t is None:
                    stacked.append(torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device))
                    continue
                if t.shape != ref_shape:
                    if t.numel() == ref.numel():
                        t = t.reshape(ref_shape)
                    else:
                        stacked.append(torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device))
                        continue
                stacked.append(t.to(device=ref_device, dtype=ref_dtype))
            return torch.stack(stacked, dim=0)
        return torch.as_tensor(x, dtype=torch.float32)

    def _maybe_detach(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return tensor.detach() if self.detach_inputs else tensor

    @staticmethod
    def _ensure_bstjc(tensor, batch_size):
        """Normalize keypoint tensor to [B, S, J, C]."""
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.dim() == 5:
            # [B, V, S, J, C] -> reduce views.
            if tensor.shape[0] == batch_size:
                tensor = tensor.mean(dim=1)
            elif tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, -1, -1, -1, -1).mean(dim=1)
            elif batch_size == 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            else:
                return None
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.shape[0] == batch_size:
                tensor = tensor.unsqueeze(1)
            elif tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.unsqueeze(1).expand(batch_size, -1, -1, -1)
            elif batch_size == 1:
                tensor = tensor.unsqueeze(0)
            else:
                return None
        elif tensor.dim() == 4:
            if tensor.shape[0] == batch_size:
                pass
            elif tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, -1, -1, -1)
            elif batch_size == 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            else:
                return None
        else:
            return None
        return tensor

    @staticmethod
    def _align_temporal_length(global_kps, kps):
        if global_kps.shape[1] == kps.shape[1]:
            return global_kps, kps
        if global_kps.shape[1] == 1 and kps.shape[1] > 1:
            return global_kps.expand(-1, kps.shape[1], -1, -1), kps
        if kps.shape[1] == 1 and global_kps.shape[1] > 1:
            return global_kps, kps.expand(-1, global_kps.shape[1], -1, -1)
        return None, None

    def _mix_pred_gt(self, pred, gt):
        if pred is None and gt is None:
            return None
        if pred is None:
            return gt
        if gt is None:
            return pred
        if not self.training or self.use_gt_ratio <= 0:
            return pred
        if self.use_gt_ratio >= 1.0:
            return gt
        if torch.rand(1).item() < self.use_gt_ratio:
            return gt
        return pred

    @staticmethod
    def _build_source_skeleton(source_format: str):
        sf = str(source_format).lower()
        if sf in {"smpl"}:
            return SMPLSkeleton
        if sf in {"mmbody"}:
            return MMBodySkeleton
        if sf in {"h36m", "mmfi"}:
            return H36MSkeleton
        if sf in {"coco"}:
            return COCOSkeleton
        if sf in {"simple_coco", "simplecoco"}:
            return SimpleCOCOSkeleton
        if sf in {"milipoint"}:
            return MiliPointSkeleton
        return None

    def _resolve_source_format(self, points, source_format: str) -> str:
        sf = str(source_format).lower()
        if sf == "simplecoco":
            sf = "simple_coco"
        if sf != "auto":
            return sf
        j = int(points.shape[-2])
        if j == SMPLSkeleton.num_joints:
            return "smpl"
        if j == MMBodySkeleton.num_joints:
            return "mmbody"
        if j == H36MSkeleton.num_joints:
            return "h36m"
        if j == COCOSkeleton.num_joints:
            return "coco"
        if j == SimpleCOCOSkeleton.num_joints:
            return "simple_coco"
        if j == MiliPointSkeleton.num_joints:
            return "milipoint"
        return "unknown"

    def _convert_to_target_skeleton(self, points, source_format: str):
        if points is None:
            return None
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, dtype=torch.float32)
        if points.dim() < 2:
            return None

        sf = self._resolve_source_format(points, source_format)
        source_skeleton = self._build_source_skeleton(sf)
        if source_skeleton is None:
            raise ValueError(
                f"Unknown source skeleton format '{sf}' (configured: '{source_format}') "
                f"for points shape {tuple(points.shape)}. "
                "Supported formats: ['smpl', 'mmbody', 'h36m', 'mmfi', 'coco', 'simple_coco', "
                "'simplecoco', 'milipoint', 'auto']."
            )

        if not hasattr(source_skeleton, "to_simple_coco"):
            raise ValueError(
                f"Skeleton format '{sf}' does not provide to_simple_coco conversion."
            )

        converted = source_skeleton.to_simple_coco(points)
        if converted.shape[-2] > self.num_joints:
            converted = converted[..., : self.num_joints, :]
        elif converted.shape[-2] < self.num_joints:
            pad_n = self.num_joints - converted.shape[-2]
            pad = torch.zeros(
                *converted.shape[:-2], pad_n, converted.shape[-1],
                device=converted.device, dtype=converted.dtype
            )
            converted = torch.cat([converted, pad], dim=-2)
        return converted

    def _projection_losses(self, pred_camera_enc_list, stream_specs, data_batch):
        losses = {}
        if len(pred_camera_enc_list) == 0:
            return losses
        gt_keypoints = data_batch.get("gt_keypoints", None)
        if gt_keypoints is None:
            return losses
        gt_keypoints = self._to_tensor(gt_keypoints)
        if gt_keypoints is None or not isinstance(gt_keypoints, torch.Tensor):
            return losses

        device = pred_camera_enc_list[0].device
        gt_keypoints = gt_keypoints.to(device).float()
        if gt_keypoints.dim() == 2:
            gt_keypoints = gt_keypoints.unsqueeze(0)

        for pred_camera_enc, (modality, sensor_idx) in zip(pred_camera_enc_list, stream_specs):
            if not self._is_branch_enabled(modality):
                continue
            modality = modality.lower()
            batch_size = pred_camera_enc.shape[0]
            gt_camera = self._get_gt_camera_encoding(
                data_batch,
                modality,
                device,
                pred_camera_enc.dtype,
                batch_size,
                sensor_idx=sensor_idx,
            )
            if gt_camera is None:
                continue

            if gt_keypoints.shape[0] == batch_size:
                gt_keypoints_b = gt_keypoints
            elif gt_keypoints.shape[0] == 1 and batch_size > 1:
                gt_keypoints_b = gt_keypoints.expand(batch_size, *gt_keypoints.shape[1:])
            elif batch_size == 1 and gt_keypoints.shape[0] > 1:
                gt_keypoints_b = gt_keypoints[-1:].contiguous()
            else:
                continue

            valid = torch.isfinite(pred_camera_enc).all(dim=-1) & torch.isfinite(gt_camera).all(dim=-1)
            valid = valid & torch.isfinite(gt_keypoints_b).view(batch_size, -1).all(dim=-1)
            if valid.sum().item() == 0:
                continue

            pred_camera_enc_v = pred_camera_enc[valid]
            gt_camera_v = gt_camera[valid]
            gt_keypoints_v = gt_keypoints_b[valid]
            if gt_keypoints_v.dim() == 4:
                gt_keypoints_v = gt_keypoints_v.reshape(
                    gt_keypoints_v.shape[0],
                    -1,
                    gt_keypoints_v.shape[-1],
                )
            if gt_keypoints_v.dim() != 3 or gt_keypoints_v.shape[-1] != 3:
                continue

            if modality in {"rgb", "depth"} and self.proj_loss_weight_rgb > 0:
                image_size = self._get_image_size(data_batch, modality)
                pred_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(pred_camera_enc_v, None)
                gt_extrinsics, gt_intrinsics = self._pose_enc_to_extrinsics_intrinsics(
                    gt_camera_v, image_size
                )
                pred_proj = self._project_to_image(gt_keypoints_v, pred_extrinsics, gt_intrinsics)
                gt_proj = self._project_to_image(gt_keypoints_v, gt_extrinsics, gt_intrinsics)
                pred_proj = self._normalize_2d(pred_proj, image_size)
                gt_proj = self._normalize_2d(gt_proj, image_size)
                pred_proj = torch.clamp(pred_proj, -1.0, 1.0)
                gt_proj = torch.clamp(gt_proj, -1.0, 1.0)
                loss_val = self._projection_loss(pred_proj, gt_proj, self.proj_loss_type)
                losses[f"proj_{modality}_s{sensor_idx}"] = (loss_val, self.proj_loss_weight_rgb)
            elif modality in {"lidar", "mmwave"} and self.proj_loss_weight_lidar > 0:
                pred_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(pred_camera_enc_v, None)
                gt_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(gt_camera_v, None)
                pred_points = self._transform_to_camera(gt_keypoints_v, pred_extrinsics)
                gt_points = self._transform_to_camera(gt_keypoints_v, gt_extrinsics)
                loss_val = self._projection_loss(pred_points, gt_points, self.proj_loss_type)
                losses[f"proj_{modality}_s{sensor_idx}"] = (loss_val, self.proj_loss_weight_lidar)

        return losses

    def _is_branch_enabled(self, modality: str) -> bool:
        modality_l = str(modality).lower()
        is_2d = modality_l in {"rgb", "depth"}
        if self.train_branch == "both":
            return True
        if self.train_branch == "2d":
            return is_2d
        return not is_2d

    def _pose_enc_to_extrinsics_intrinsics(self, pose_enc, image_size_hw):
        pose_enc = pose_enc.unsqueeze(1)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=image_size_hw,
            pose_encoding_type=self.pose_encoding_type,
            build_intrinsics=image_size_hw is not None,
        )
        return extrinsics.squeeze(1), None if intrinsics is None else intrinsics.squeeze(1)

    def _get_gt_camera_encoding(self, data_batch, modality, device, dtype, batch_size, sensor_idx=0):
        gt_camera = data_batch.get(f"gt_camera_{modality}", None)
        if gt_camera is None:
            return None
        if not isinstance(gt_camera, torch.Tensor):
            gt_camera = torch.as_tensor(gt_camera, dtype=dtype)
        gt_camera = gt_camera.to(device=device, dtype=dtype)

        if gt_camera.dim() == 4:  # B V S C
            if gt_camera.shape[-1] != self.pose_encoding_dim:
                return None
            view_idx = min(sensor_idx, gt_camera.shape[1] - 1)
            return gt_camera[:, view_idx, -1, :]
        if gt_camera.dim() == 3:
            if gt_camera.shape[-1] != self.pose_encoding_dim:
                return None
            if gt_camera.shape[0] == batch_size:
                return gt_camera[:, -1, :]
            if batch_size == 1:
                view_idx = min(sensor_idx, gt_camera.shape[0] - 1)
                return gt_camera[view_idx : view_idx + 1, -1, :]
            return None
        if gt_camera.dim() == 2:
            if gt_camera.shape[-1] != self.pose_encoding_dim:
                return None
            if gt_camera.shape[0] == batch_size:
                return gt_camera
            if batch_size == 1:
                view_idx = min(sensor_idx, gt_camera.shape[0] - 1)
                return gt_camera[view_idx : view_idx + 1, :]
            return None
        if gt_camera.dim() == 1:
            if gt_camera.shape[0] != self.pose_encoding_dim:
                return None
            gt_camera = gt_camera.unsqueeze(0)
            if batch_size > 1:
                gt_camera = gt_camera.expand(batch_size, -1)
            return gt_camera
        return None

    def _get_image_size(self, data_batch, modality):
        size_key = f"image_size_hw_{modality}"
        if size_key in data_batch:
            size = data_batch[size_key]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
            if isinstance(size, (list, tuple)) and len(size) > 0 and isinstance(size[0], (list, tuple)) and len(size[0]) == 2:
                return int(size[0][0]), int(size[0][1])
        if "image_size_hw" in data_batch:
            size = data_batch["image_size_hw"]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
            if isinstance(size, (list, tuple)) and len(size) > 0 and isinstance(size[0], (list, tuple)) and len(size[0]) == 2:
                return int(size[0][0]), int(size[0][1])
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
        cam_points = KeypointCameraGCNHeadV5._transform_to_camera(points, extrinsics)
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

    @staticmethod
    def _projection_loss(pred, target, loss_type="l1"):
        if loss_type.lower() in {"l1", "mae"}:
            loss = F.l1_loss(pred, target)
            return KeypointCameraGCNHeadV5._sanitize_loss(loss)
        if loss_type.lower() in {"l2", "mse"}:
            loss = F.mse_loss(pred, target)
            return KeypointCameraGCNHeadV5._sanitize_loss(loss)
        raise ValueError(f"Unsupported projection loss type: {loss_type}")

    @staticmethod
    def _sanitize_loss(loss):
        if loss is None:
            return loss
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros_like(loss)
        return loss
