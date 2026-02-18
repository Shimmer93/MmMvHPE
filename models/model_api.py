import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import os
import pickle
from einops import rearrange
import matplotlib.pyplot as plt
import wandb

from misc.registry import create_model, create_metric, create_optimizer, create_scheduler
from misc.vis import visualize_multimodal_sample
from misc.utils import torch2numpy, load_state_dict_part
from misc.camera_batch import collect_gt_camera_encodings

class LitModel(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.camera_only = getattr(hparams, "camera_only", False)
        
        if not self.camera_only:
            assert self.with_rgb or self.with_depth or self.with_lidar or self.with_mmwave, "At least one modality should be used."

        def _cfg_eval(cfg):
            return bool(cfg.get('eval', False))

        if self.with_rgb and not self.camera_only:
            self.backbone_rgb = create_model(
                self.hparams.backbone_rgb['name'],
                self.hparams.backbone_rgb['params'],
                eval=_cfg_eval(self.hparams.backbone_rgb),
            )
            self.has_temporal_rgb = self.hparams.backbone_rgb['has_temporal']
            if hasattr(self.hparams, 'pretrained_rgb_path'):
                print("Loading pretrained RGB backbone from:", self.hparams.pretrained_rgb_path)
                state_dict_rgb = torch.load(self.hparams.pretrained_rgb_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_rgb, state_dict_rgb, prefix='backbone_rgb.')
        if self.with_depth and not self.camera_only:
            self.backbone_depth = create_model(
                self.hparams.backbone_depth['name'],
                self.hparams.backbone_depth['params'],
                eval=_cfg_eval(self.hparams.backbone_depth),
            )
            self.has_temporal_depth = self.hparams.backbone_depth['has_temporal']
            if hasattr(self.hparams, 'pretrained_depth_path'):
                print("Loading pretrained Depth backbone from:", self.hparams.pretrained_depth_path)
                state_dict_depth = torch.load(self.hparams.pretrained_depth_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_depth, state_dict_depth, prefix='backbone_depth.')
        if self.with_lidar and not self.camera_only:
            self.backbone_lidar = create_model(
                self.hparams.backbone_lidar['name'],
                self.hparams.backbone_lidar['params'],
                eval=_cfg_eval(self.hparams.backbone_lidar),
            )
            self.has_temporal_lidar = self.hparams.backbone_lidar['has_temporal']
            if hasattr(self.hparams, 'pretrained_lidar_path'):
                print("Loading pretrained LIDAR backbone from:", self.hparams.pretrained_lidar_path)
                state_dict_lidar = torch.load(self.hparams.pretrained_lidar_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_lidar, state_dict_lidar, prefix='backbone_lidar.')
        if self.with_mmwave and not self.camera_only:
            self.backbone_mmwave = create_model(
                self.hparams.backbone_mmwave['name'],
                self.hparams.backbone_mmwave['params'],
                eval=_cfg_eval(self.hparams.backbone_mmwave),
            )
            self.has_temporal_mmwave = self.hparams.backbone_mmwave['has_temporal']
            if hasattr(self.hparams, 'pretrained_mmwave_path'):
                print("Loading pretrained MMWave backbone from:", self.hparams.pretrained_mmwave_path)
                state_dict_mmwave = torch.load(self.hparams.pretrained_mmwave_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_mmwave, state_dict_mmwave, prefix='backbone_mmwave.')

        if not self.camera_only:
            self.aggregator = create_model(
                self.hparams.aggregator['name'],
                self.hparams.aggregator['params'],
                eval=_cfg_eval(self.hparams.aggregator),
            )
        else:
            self.aggregator = None

        if self.with_keypoint_head:
            self.keypoint_head = create_model(
                self.hparams.keypoint_head['name'],
                self.hparams.keypoint_head['params'],
                eval=_cfg_eval(self.hparams.keypoint_head),
            )
        if self.with_smpl_head:
            self.smpl_head = create_model(
                self.hparams.smpl_head['name'],
                self.hparams.smpl_head['params'],
                eval=_cfg_eval(self.hparams.smpl_head),
            )
        if self.with_camera_head:
            self.camera_head = create_model(
                self.hparams.camera_head['name'],
                self.hparams.camera_head['params'],
                eval=_cfg_eval(self.hparams.camera_head),
            )
        
        if hparams.checkpoint_path is not None:
            print("Loading model checkpoint from:", hparams.checkpoint_path)
            state_dict = torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict']
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('camera_head.')}
            self.load_state_dict(state_dict, strict=False)

        if hasattr(hparams, 'pretrained_camera_head_path') and self.with_camera_head:
            print("Loading pretrained Camera Head from:", hparams.pretrained_camera_head_path)
            state_dict_camera_head = torch.load(hparams.pretrained_camera_head_path, map_location=self.device)['state_dict']
            load_state_dict_part(self.camera_head, state_dict_camera_head, prefix='camera_head.')

        self.metrics = {metric['alias' if 'alias' in metric else 'name']: create_metric(metric['name'], metric['params']) for metric in hparams.metrics}

        if self.hparams.save_test_preds:
            self.test_preds = []

    @property
    def with_rgb(self):
        return hasattr(self.hparams, 'backbone_rgb')
    
    @property
    def with_depth(self):
        return hasattr(self.hparams, 'backbone_depth')
    
    @property
    def with_lidar(self):
        return hasattr(self.hparams, 'backbone_lidar')
    
    @property
    def with_mmwave(self):
        return hasattr(self.hparams, 'backbone_mmwave')
    
    @property
    def with_camera_head(self):
        return hasattr(self.hparams, 'camera_head')
    
    @property
    def with_keypoint_head(self):
        return hasattr(self.hparams, 'keypoint_head') and not self.camera_only
    
    @property
    def with_smpl_head(self):
        return hasattr(self.hparams, 'smpl_head') and not self.camera_only

    @staticmethod
    def _flatten_input_views(frames, expected_dims_without_view):
        if frames is None:
            return None, 1
        if frames.dim() == expected_dims_without_view:
            return frames, 1
        if frames.dim() == expected_dims_without_view + 1:
            b, v = frames.shape[:2]
            flat = frames.reshape(b * v, *frames.shape[2:])
            return flat, v
        raise ValueError(
            f"Unexpected input shape {tuple(frames.shape)}. "
            f"Expected {expected_dims_without_view}D or {expected_dims_without_view + 1}D tensor."
        )

    @staticmethod
    def _restore_view_features(features, batch_size, num_views):
        if num_views == 1:
            return features
        return features.reshape(batch_size, num_views, *features.shape[1:])
    
    def forward_rgb(self, frames_rgb):
        """Forward function for RGB frames."""
        if not self.with_rgb or frames_rgb is None:
            return None
        
        B = frames_rgb.shape[0]
        frames_rgb, num_views = self._flatten_input_views(frames_rgb, expected_dims_without_view=5)
        if self.has_temporal_rgb:
            features_rgb = self.backbone_rgb(frames_rgb)
        else:
            frames_rgb = rearrange(frames_rgb, 'b t ... -> (b t) ...')
            features_rgb = self.backbone_rgb(frames_rgb)
            features_rgb = rearrange(features_rgb, '(b t) ... -> b t ...', b=B * num_views)
        features_rgb = self._restore_view_features(features_rgb, batch_size=B, num_views=num_views)

        return features_rgb
    
    def forward_depth(self, frames_depth):
        """Forward function for depth frames."""
        if not self.with_depth or frames_depth is None:
            return None

        B = frames_depth.shape[0]
        frames_depth, num_views = self._flatten_input_views(frames_depth, expected_dims_without_view=5)
        if self.has_temporal_depth:
            features_depth = self.backbone_depth(frames_depth)
        else:
            frames_depth = rearrange(frames_depth, 'b t ... -> (b t) ...')
            features_depth = self.backbone_depth(frames_depth)
            features_depth = rearrange(features_depth, '(b t) ... -> b t ...', b=B * num_views)
        features_depth = self._restore_view_features(features_depth, batch_size=B, num_views=num_views)

        return features_depth
    
    def forward_lidar(self, frames_lidar):
        """Forward function for LiDAR frames."""
        if not self.with_lidar or frames_lidar is None:
            return None

        B = frames_lidar.shape[0]
        frames_lidar, num_views = self._flatten_input_views(frames_lidar, expected_dims_without_view=4)
        if self.has_temporal_lidar:
            features_lidar = self.backbone_lidar(frames_lidar)
        else:
            frames_lidar = rearrange(frames_lidar, 'b t n c -> (b t) n c')
            features_lidar = self.backbone_lidar(frames_lidar)
            features_lidar = rearrange(features_lidar, '(b t) ... -> b t ...', b=B * num_views)
        features_lidar = self._restore_view_features(features_lidar, batch_size=B, num_views=num_views)

        return features_lidar
    
    def forward_mmwave(self, frames_mmwave):
        """Forward function for mmWave frames."""
        if not self.with_mmwave or frames_mmwave is None:
            return None
        
        B = frames_mmwave.shape[0]
        frames_mmwave, num_views = self._flatten_input_views(frames_mmwave, expected_dims_without_view=4)
        if self.has_temporal_mmwave:
            features_mmwave = self.backbone_mmwave(frames_mmwave)
        else:
            frames_mmwave = rearrange(frames_mmwave, 'b t n c -> (b t) n c')
            features_mmwave = self.backbone_mmwave(frames_mmwave)
            features_mmwave = rearrange(features_mmwave, '(b t) ... -> b t ...', b=B * num_views)
        features_mmwave = self._restore_view_features(features_mmwave, batch_size=B, num_views=num_views)

        return features_mmwave

    def extract_features(self, batch):
        if self.camera_only:
            return None
        feat_rgb = self.forward_rgb(batch.get('input_rgb', None))
        feat_depth = self.forward_depth(batch.get('input_depth', None))
        feat_lidar = self.forward_lidar(batch.get('input_lidar', None))
        feat_mmwave = self.forward_mmwave(batch.get('input_mmwave', None))
        return feat_rgb, feat_depth, feat_lidar, feat_mmwave
    
    def aggregate_features(self, feats, batch):
        if self.camera_only:
            return None
        return self.aggregator(feats, **batch)

    def training_step(self, batch, batch_idx):
        feats = self.extract_features(batch)
        feats_agg = self.aggregate_features(feats, batch)
        loss_dict = {}
        log_dict = {}

        if self.camera_only and self.with_camera_head:
            losses_camera = self.camera_head.loss(feats_agg, batch, pred_dict={})
            loss_dict.update(losses_camera)
        else:
            if self.with_keypoint_head:
                losses_keypoint = self.keypoint_head.loss(feats_agg, batch)
                loss_dict.update(losses_keypoint)

            if self.with_smpl_head:
                losses_smpl = self.smpl_head.loss(feats_agg, batch)
                loss_dict.update(losses_smpl)

            if self.with_camera_head:
                pred_dict = {}
                if self.with_keypoint_head:
                    with torch.no_grad():
                        preds_keypoint = self.keypoint_head.predict(feats_agg)
                        pred_dict["pred_keypoints"] = preds_keypoint
                        self._attach_keypoint_modalities(pred_dict, feats_agg, batch)
                losses_camera = self.camera_head.loss(feats_agg, batch, pred_dict=pred_dict)
                loss_dict.update(losses_camera)

        loss = 0
        for loss_name, (loss_value, loss_weight) in loss_dict.items():
            loss += loss_value * loss_weight
            log_dict[f'train_{loss_name}'] = loss_value

        log_dict['train_loss'] = loss

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        feats = self.extract_features(batch)
        feats_agg = self.aggregate_features(feats, batch)

        pred_dict = {}
        if self.camera_only and self.with_camera_head:
            preds_camera = self._predict_camera(feats_agg, batch, pred_dict)
            self._attach_camera_predictions(pred_dict, preds_camera, batch)
        else:
            if self.with_keypoint_head:
                preds_keypoint = self.keypoint_head.predict(feats_agg)
                pred_dict['pred_keypoints'] = preds_keypoint
            if self.with_smpl_head:
                preds_smpl = self.smpl_head.predict(feats_agg)
                pred_dict['pred_smpl_params'] = preds_smpl['pred_smpl_params']
                pred_dict['pred_smpl_keypoints'] = preds_smpl['pred_keypoints']
                if 'pred_keypoints' in preds_smpl and 'pred_keypoints' not in pred_dict:
                    pred_dict['pred_keypoints'] = preds_smpl['pred_keypoints']
                if 'pred_smpl' in preds_smpl:
                    pred_dict['pred_smpl'] = preds_smpl['pred_smpl']
            self._attach_keypoint_modalities(pred_dict, feats_agg, batch)
            if self.with_camera_head:
                preds_camera = self._predict_camera(feats_agg, batch, pred_dict)
                self._attach_camera_predictions(pred_dict, preds_camera, batch)

        log_dict = {}
        for _, metric in self.metrics.items():
            log_dict[f'val_{metric.name}'] = metric(pred_dict, batch)

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self.visualize(batch, pred_dict, stage="val")

    def test_step(self, batch, batch_idx):
        feats = self.extract_features(batch)
        feats_agg = self.aggregate_features(feats, batch)

        pred_dict = {}
        if self.camera_only and self.with_camera_head:
            preds_camera = self._predict_camera(feats_agg, batch, pred_dict)
            self._attach_camera_predictions(pred_dict, preds_camera, batch)
        else:
            if self.with_keypoint_head:
                preds_keypoint = self.keypoint_head.predict(feats_agg)
                pred_dict['pred_keypoints'] = preds_keypoint
            if self.with_smpl_head:
                preds_smpl = self.smpl_head.predict(feats_agg)
                pred_dict['pred_smpl_params'] = preds_smpl['pred_smpl_params']
                pred_dict['pred_smpl_keypoints'] = preds_smpl['pred_keypoints']
                if 'pred_keypoints' in preds_smpl and 'pred_keypoints' not in pred_dict:
                    pred_dict['pred_keypoints'] = preds_smpl['pred_keypoints']
                if 'pred_smpl' in preds_smpl:
                    pred_dict['pred_smpl'] = preds_smpl['pred_smpl']
            self._attach_keypoint_modalities(pred_dict, feats_agg, batch)
            if self.with_camera_head:
                preds_camera = self._predict_camera(feats_agg, batch, pred_dict)
                self._attach_camera_predictions(pred_dict, preds_camera, batch)

        log_dict = {}
        for _, metric in self.metrics.items():
            log_dict[f'test_{metric.name}'] = metric(pred_dict, batch)

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        num_batches = sum(self.trainer.num_test_batches) # list of ints
        visualize_interval = max(1, num_batches // 5)
        if batch_idx in range(0, num_batches, visualize_interval):
            self.visualize(batch, pred_dict, stage="test", batch_idx=batch_idx)
    
        if self.hparams.save_test_preds:
            batch_size = len(batch['sample_id'])
            pose_encoding_dim = int(getattr(getattr(self, "camera_head", None), "pose_encoding_dim", 9))
            modalities = self._normalize_modalities(batch.get("modalities", []))
            gt_cameras_batch = collect_gt_camera_encodings(
                data_batch=batch,
                modalities=modalities,
                batch_size=batch_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
                pose_encoding_dim=pose_encoding_dim,
            )
            stream_specs = []
            pred_cameras_stream = None
            gt_cameras_stream = None
            if 'pred_cameras_stream' in pred_dict or 'pred_cameras' in pred_dict:
                pred_source = pred_dict.get('pred_cameras_stream', pred_dict.get('pred_cameras'))
                pred_cameras_stream = self._select_camera_tensor(pred_source)
            if isinstance(pred_cameras_stream, torch.Tensor):
                if pred_cameras_stream.dim() != 3:
                    raise ValueError(
                        f"Expected stream cameras as [B,S,D], got {tuple(pred_cameras_stream.shape)}."
                    )
                stream_specs = self._build_stream_specs(batch, target_streams=pred_cameras_stream.shape[1])
                gt_cameras_stream = self._collect_gt_cameras_stream(
                    batch=batch,
                    stream_specs=stream_specs,
                    batch_size=batch_size,
                    pose_encoding_dim=pose_encoding_dim,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
            for i in range(batch_size):
                self.test_preds.append({
                    'sample_id': batch['sample_id'][i],
                    'camera_modalities': modalities,
                    'camera_stream_modalities': [m for m, _ in stream_specs] if stream_specs else None,
                    'camera_stream_sensor_indices': [s for _, s in stream_specs] if stream_specs else None,
                    'pred_cameras': torch2numpy(pred_dict['pred_cameras'][i]) if 'pred_cameras' in pred_dict else None,
                    'gt_cameras': torch2numpy(gt_cameras_batch[i]) if gt_cameras_batch is not None else None,
                    'pred_cameras_stream': torch2numpy(pred_cameras_stream[i]) if pred_cameras_stream is not None else None,
                    'gt_cameras_stream': torch2numpy(gt_cameras_stream[i]) if gt_cameras_stream is not None else None,
                    'pred_keypoints': torch2numpy(pred_dict['pred_keypoints'][i]) if 'pred_keypoints' in pred_dict else None,
                    'pred_smpl_params': torch2numpy(pred_dict['pred_smpl_params'][i]) if 'pred_smpl_params' in pred_dict else None,
                    'pred_smpl_keypoints': torch2numpy(pred_dict['pred_smpl_keypoints'][i]) if 'pred_smpl_keypoints' in pred_dict else None,
                    'gt_smpl_params': torch2numpy(batch['gt_smpl_params'][i]) if 'gt_smpl_params' in batch else None,
                    'gt_keypoints': torch2numpy(batch['gt_keypoints'][i]) if 'gt_keypoints' in batch else None
                })

    def predict_step(self, batch, batch_idx):
        feats = self.extract_features(batch)
        feats_agg = self.aggregate_features(feats, batch)

        pred_dict = {}
        if self.camera_only and self.with_camera_head:
            preds_camera = self._predict_camera(feats_agg, batch, pred_dict)
            self._attach_camera_predictions(pred_dict, preds_camera, batch)
        else:
            if self.with_keypoint_head:
                preds_keypoint = self.keypoint_head.predict(feats_agg)
                pred_dict['pred_keypoints'] = preds_keypoint
            if self.with_smpl_head:
                preds_smpl = self.smpl_head.predict(feats_agg)
                pred_dict['pred_smpl_params'] = preds_smpl['pred_smpl_params']
                pred_dict['pred_smpl_keypoints'] = preds_smpl['pred_keypoints']
                if 'pred_keypoints' in preds_smpl and 'pred_keypoints' not in pred_dict:
                    pred_dict['pred_keypoints'] = preds_smpl['pred_keypoints']
                if 'pred_smpl' in preds_smpl:
                    pred_dict['pred_smpl'] = preds_smpl['pred_smpl']
            self._attach_keypoint_modalities(pred_dict, feats_agg, batch)
            if self.with_camera_head:
                preds_camera = self._predict_camera(feats_agg, batch, pred_dict)
                self._attach_camera_predictions(pred_dict, preds_camera, batch)

        return pred_dict

    def _predict_camera(self, feats_agg, batch, pred_dict):
        return self.camera_head.predict(feats_agg, data_batch=batch, pred_dict=pred_dict)

    @staticmethod
    def _normalize_modalities(modalities):
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = list(modalities[0])
        elif isinstance(modalities, (list, tuple)):
            modalities = list(modalities)
        else:
            raise ValueError(f"Expected `modalities` as list/tuple, got {type(modalities).__name__}.")
        if len(modalities) == 0:
            raise ValueError("`modalities` is empty.")
        return [str(m).lower() for m in modalities]

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

    def _infer_sensor_count(self, batch, modality):
        if not isinstance(batch, dict):
            raise ValueError("Batch must be a dict.")
        selected = batch.get("selected_cameras", None)
        if isinstance(selected, (list, tuple)):
            if len(selected) == 0:
                raise ValueError("`selected_cameras` list is empty.")
            selected = selected[0]
        if not isinstance(selected, dict):
            raise ValueError(
                "Expected `selected_cameras` to be a dict (or list containing one dict) in batch."
            )

        cams = selected.get(modality, None)
        if not isinstance(cams, (list, tuple)) or len(cams) == 0:
            raise ValueError(
                f"`selected_cameras[{modality}]` must be a non-empty list/tuple, got {type(cams).__name__}."
            )
        return len(cams)

    def _build_stream_specs(self, batch, target_streams=None):
        modalities = self._normalize_modalities(batch.get("modalities", []))
        specs = []
        for modality in modalities:
            n_sensors = self._infer_sensor_count(batch, modality)
            for sensor_idx in range(max(1, int(n_sensors))):
                specs.append((modality, sensor_idx))

        if target_streams is not None:
            target_streams = int(max(0, target_streams))
            if len(specs) != target_streams:
                raise ValueError(
                    f"Stream count mismatch: inferred {len(specs)} from batch metadata, "
                    f"but target_streams={target_streams}."
                )
        return specs

    @staticmethod
    def _select_camera_tensor(pred_cameras):
        if isinstance(pred_cameras, list):
            if len(pred_cameras) == 0:
                raise ValueError("pred_cameras list is empty.")
            tensor_items = [x for x in pred_cameras if isinstance(x, torch.Tensor)]
            if len(tensor_items) != len(pred_cameras):
                raise ValueError("pred_cameras list contains non-tensor items.")
            return tensor_items[-1]
        if isinstance(pred_cameras, torch.Tensor):
            return pred_cameras
        raise ValueError(
            f"Expected pred_cameras as Tensor or list[Tensor], got {type(pred_cameras).__name__}."
        )

    def _reduce_stream_cameras_for_metrics(self, pred_cameras, batch):
        if isinstance(pred_cameras, list):
            return [self._reduce_stream_cameras_for_metrics(x, batch) for x in pred_cameras]
        if not isinstance(pred_cameras, torch.Tensor):
            raise ValueError(
                f"Expected pred_cameras tensor in reducer, got {type(pred_cameras).__name__}."
            )
        if pred_cameras.dim() != 3:
            raise ValueError(
                f"Expected pred_cameras with shape [B,S,D], got {tuple(pred_cameras.shape)}."
            )

        modalities = self._normalize_modalities(batch.get("modalities", []))

        stream_specs = self._build_stream_specs(batch, target_streams=pred_cameras.shape[1])

        out = torch.full(
            (pred_cameras.shape[0], len(modalities), pred_cameras.shape[-1]),
            float("nan"),
            device=pred_cameras.device,
            dtype=pred_cameras.dtype,
        )
        for m_idx, modality in enumerate(modalities):
            idxs = [i for i, (mod, _) in enumerate(stream_specs) if mod == modality]
            if len(idxs) == 0:
                raise ValueError(f"No sensor streams found for modality `{modality}`.")
            out[:, m_idx] = pred_cameras[:, idxs, :].mean(dim=1)
        return out

    def _attach_camera_predictions(self, pred_dict, preds_camera, batch):
        pred_dict["pred_cameras_stream"] = preds_camera
        pred_dict["pred_cameras"] = self._reduce_stream_cameras_for_metrics(preds_camera, batch)

    @staticmethod
    def _select_gt_camera_sensor(gt_camera, batch_size, sensor_idx, pose_encoding_dim, device, dtype):
        if gt_camera is None:
            raise ValueError("gt_camera is None.")
        if not isinstance(gt_camera, torch.Tensor):
            gt_camera = torch.as_tensor(gt_camera, dtype=dtype)
        gt_camera = gt_camera.to(device=device, dtype=dtype)
        if gt_camera.shape[-1] != pose_encoding_dim:
            raise ValueError(
                f"Expected last dim {pose_encoding_dim}, got shape {tuple(gt_camera.shape)}."
            )

        if gt_camera.dim() == 4:  # [B, V, S, D]
            if gt_camera.shape[0] != batch_size:
                raise ValueError(
                    f"Expected gt_camera batch {batch_size}, got {gt_camera.shape[0]} for shape {tuple(gt_camera.shape)}."
                )
            if sensor_idx >= gt_camera.shape[1]:
                raise ValueError(
                    f"sensor_idx={sensor_idx} out of range for V={gt_camera.shape[1]}."
                )
            return gt_camera[:, sensor_idx, -1, :]

        if gt_camera.dim() == 3:  # [B, S, D]
            if gt_camera.shape[0] != batch_size:
                raise ValueError(
                    f"Expected gt_camera shape [B,S,D] with B={batch_size}, got {tuple(gt_camera.shape)}."
                )
            if sensor_idx != 0:
                raise ValueError(
                    f"Single-sensor gt_camera received sensor_idx={sensor_idx}."
                )
            return gt_camera[:, -1, :]

        if gt_camera.dim() == 2:  # [B, D]
            if gt_camera.shape[0] != batch_size:
                raise ValueError(
                    f"Expected gt_camera shape [B,D] with B={batch_size}, got {tuple(gt_camera.shape)}."
                )
            if sensor_idx != 0:
                raise ValueError(
                    f"Single-sensor gt_camera received sensor_idx={sensor_idx}."
                )
            return gt_camera

        raise ValueError(
            f"Unsupported gt_camera shape {tuple(gt_camera.shape)}; expected [B,V,S,D], [B,S,D], or [B,D]."
        )

    def _collect_gt_cameras_stream(
        self,
        batch,
        stream_specs,
        batch_size,
        pose_encoding_dim,
        device,
        dtype,
    ):
        if len(stream_specs) == 0:
            raise ValueError("stream_specs is empty.")
        out = torch.full(
            (batch_size, len(stream_specs), pose_encoding_dim),
            float("nan"),
            device=device,
            dtype=dtype,
        )
        for s_idx, (modality, sensor_idx) in enumerate(stream_specs):
            gt = self._select_gt_camera_sensor(
                batch.get(f"gt_camera_{modality}", None),
                batch_size=batch_size,
                sensor_idx=sensor_idx,
                pose_encoding_dim=pose_encoding_dim,
                device=device,
                dtype=dtype,
            )
            out[:, s_idx] = gt
        return out

    def _attach_keypoint_modalities(self, pred_dict, feats_agg, batch):
        if not self.with_keypoint_head:
            return
        modalities = self._normalize_modalities(batch.get("modalities", []))
        try:
            outputs = self.keypoint_head.forward(feats_agg, modalities=modalities, data_batch=batch)
        except TypeError:
            try:
                outputs = self.keypoint_head.forward(feats_agg, modalities=modalities)
            except TypeError:
                return
        if not isinstance(outputs, dict):
            return
        per_modality = outputs.get("per_modality")
        if per_modality is None:
            return

        if isinstance(per_modality, torch.Tensor):
            if per_modality.dim() >= 4:
                per_stream = [per_modality[:, i] for i in range(per_modality.shape[1])]
            else:
                per_stream = [per_modality[i] for i in range(per_modality.shape[0])]
        elif isinstance(per_modality, (list, tuple)):
            per_stream = list(per_modality)
        else:
            return

        stream_modalities = outputs.get("stream_modalities", None)
        stream_sensor_indices = outputs.get("stream_sensor_indices", None)
        if not isinstance(stream_modalities, (list, tuple)) or not isinstance(stream_sensor_indices, (list, tuple)):
            stream_specs = self._build_stream_specs(batch, target_streams=len(per_stream))
            stream_modalities = [m for m, _ in stream_specs]
            stream_sensor_indices = [s for _, s in stream_specs]

        if not (len(per_stream) == len(stream_modalities) == len(stream_sensor_indices)):
            raise ValueError(
                "Per-stream keypoint outputs and stream metadata have inconsistent lengths: "
                f"{len(per_stream)} vs {len(stream_modalities)} vs {len(stream_sensor_indices)}."
            )
        for i in range(len(per_stream)):
            mod = str(stream_modalities[i]).lower()
            sensor_idx = int(stream_sensor_indices[i])
            pred = per_stream[i]
            if mod in {"rgb", "depth"}:
                pred_dict[f"pred_keypoints_2d_{mod}_s{sensor_idx}"] = pred
                if sensor_idx == 0 and f"pred_keypoints_2d_{mod}" not in pred_dict:
                    pred_dict[f"pred_keypoints_2d_{mod}"] = pred
            else:
                pred_dict[f"pred_keypoints_3d_{mod}_s{sensor_idx}"] = pred
                if sensor_idx == 0 and f"pred_keypoints_3d_{mod}" not in pred_dict:
                    pred_dict[f"pred_keypoints_3d_{mod}"] = pred
                if mod == "lidar":
                    pred_dict[f"pred_keypoints_pc_centered_input_lidar_s{sensor_idx}"] = pred
                    if sensor_idx == 0 and "pred_keypoints_pc_centered_input_lidar" not in pred_dict:
                        pred_dict["pred_keypoints_pc_centered_input_lidar"] = pred

    def on_test_epoch_end(self):
        if not self.hparams.save_test_preds:
            return

        # save local preds to tmp and gather later
        tmp_path = os.path.join(self.trainer.default_root_dir, f'tmp_test_preds_rank_{self.global_rank}.pkl')
        with open(tmp_path, 'wb') as f:
            pickle.dump(self.test_preds, f)
        self.test_preds = []  # free memory

        # Synchronize all processes
        if hasattr(self.trainer.strategy, 'barrier'):
            self.trainer.strategy.barrier()

        if self.global_rank == 0:
            def _stack_or_object(values):
                if len(values) == 0:
                    return None
                if all(v is None for v in values):
                    return None
                if all(v is not None for v in values):
                    try:
                        return np.stack(values)
                    except Exception:
                        return np.array(values, dtype=object)
                return np.array(values, dtype=object)

            # gather all preds
            gathered_preds = []
            for rank in range(self.trainer.world_size):
                tmp_path = os.path.join(self.trainer.default_root_dir, f'tmp_test_preds_rank_{rank}.pkl')
                with open(tmp_path, 'rb') as f:
                    preds_rank = pickle.load(f)
                    gathered_preds.extend(preds_rank)
                os.remove(tmp_path)  # clean up

            # Build final predictions dictionary
            final_preds = {'sample_ids': [item['sample_id'] for item in gathered_preds]}
            
            # Robust stacking for potentially partial/missing fields across ranks/samples.
            final_preds['pred_cameras'] = _stack_or_object([item.get('pred_cameras') for item in gathered_preds])
            final_preds['gt_cameras'] = _stack_or_object([item.get('gt_cameras') for item in gathered_preds])
            final_preds['pred_cameras_stream'] = _stack_or_object([item.get('pred_cameras_stream') for item in gathered_preds])
            final_preds['gt_cameras_stream'] = _stack_or_object([item.get('gt_cameras_stream') for item in gathered_preds])
            final_preds['pred_keypoints'] = _stack_or_object([item.get('pred_keypoints') for item in gathered_preds])
            final_preds['gt_keypoints'] = _stack_or_object([item.get('gt_keypoints') for item in gathered_preds])
            final_preds['pred_smpl_params'] = _stack_or_object([item.get('pred_smpl_params') for item in gathered_preds])
            final_preds['pred_smpl_keypoints'] = _stack_or_object([item.get('pred_smpl_keypoints') for item in gathered_preds])
            final_preds['gt_smpl_params'] = _stack_or_object([item.get('gt_smpl_params') for item in gathered_preds])
            camera_modalities_list = [item.get('camera_modalities', None) for item in gathered_preds]
            if camera_modalities_list and all(m == camera_modalities_list[0] for m in camera_modalities_list):
                final_preds['camera_modalities'] = camera_modalities_list[0]
            else:
                final_preds['camera_modalities_per_sample'] = camera_modalities_list
            camera_stream_modalities_list = [item.get('camera_stream_modalities', None) for item in gathered_preds]
            if camera_stream_modalities_list and all(m == camera_stream_modalities_list[0] for m in camera_stream_modalities_list):
                final_preds['camera_stream_modalities'] = camera_stream_modalities_list[0]
            else:
                final_preds['camera_stream_modalities_per_sample'] = camera_stream_modalities_list
            camera_stream_sensor_indices_list = [item.get('camera_stream_sensor_indices', None) for item in gathered_preds]
            if camera_stream_sensor_indices_list and all(m == camera_stream_sensor_indices_list[0] for m in camera_stream_sensor_indices_list):
                final_preds['camera_stream_sensor_indices'] = camera_stream_sensor_indices_list[0]
            else:
                final_preds['camera_stream_sensor_indices_per_sample'] = camera_stream_sensor_indices_list

            # sort by sample_ids
            sorted_indices = np.argsort(final_preds['sample_ids'])
            final_preds['sample_ids'] = [final_preds['sample_ids'][i] for i in sorted_indices]
            
            for key in [
                'pred_cameras',
                'gt_cameras',
                'pred_cameras_stream',
                'gt_cameras_stream',
                'pred_keypoints',
                'gt_keypoints',
                'pred_smpl_params',
                'pred_smpl_keypoints',
                'gt_smpl_params',
            ]:
                if final_preds[key] is not None:
                    final_preds[key] = final_preds[key][sorted_indices]
            if 'camera_modalities_per_sample' in final_preds:
                final_preds['camera_modalities_per_sample'] = [
                    final_preds['camera_modalities_per_sample'][i] for i in sorted_indices
                ]
            if 'camera_stream_modalities_per_sample' in final_preds:
                final_preds['camera_stream_modalities_per_sample'] = [
                    final_preds['camera_stream_modalities_per_sample'][i] for i in sorted_indices
                ]
            if 'camera_stream_sensor_indices_per_sample' in final_preds:
                final_preds['camera_stream_sensor_indices_per_sample'] = [
                    final_preds['camera_stream_sensor_indices_per_sample'][i] for i in sorted_indices
                ]

            save_path = os.path.join('logs', self.hparams.exp_name, self.hparams.version, f'{self.hparams.model_name}_test_predictions.pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(final_preds, f)
            print(f'Saved test predictions to {save_path}')

    def visualize(self, batch, pred_dict, stage="val", batch_idx=None):
        skl_format = self.hparams.vis_skl_format if hasattr(self.hparams, 'vis_skl_format') else None
        vis_denorm_params = self.hparams.vis_denorm_params if hasattr(self.hparams, 'vis_denorm_params') else None
        smpl_path = getattr(self.hparams, 'smpl_path', None)
        if smpl_path is None and hasattr(self.hparams, 'smpl_head'):
            smpl_path = self.hparams.smpl_head.get('params', {}).get('smpl_path', None)

        fig = visualize_multimodal_sample(batch, pred_dict, skl_format, vis_denorm_params, smpl_model_path=smpl_path)

        if batch_idx is None:
            tag = f"{stage}_visualization"
        else:
            tag = f"{stage}_visualization/batch_{batch_idx}"

        if self.hparams.use_wandb:
            self.logger.log_image(key=tag, images=[fig])
        else:
            self.logger.experiment.add_figure(tag, fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]
