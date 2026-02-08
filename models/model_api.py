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

class LitModel(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.camera_only = getattr(hparams, "camera_only", False)
        
        if not self.camera_only:
            assert self.with_rgb or self.with_depth or self.with_lidar or self.with_mmwave, "At least one modality should be used."

        if self.with_rgb and not self.camera_only:
            self.backbone_rgb = create_model(self.hparams.backbone_rgb['name'], self.hparams.backbone_rgb['params'])
            self.has_temporal_rgb = self.hparams.backbone_rgb['has_temporal']
            if hasattr(self.hparams, 'pretrained_rgb_path'):
                print("Loading pretrained RGB backbone from:", self.hparams.pretrained_rgb_path)
                state_dict_rgb = torch.load(self.hparams.pretrained_rgb_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_rgb, state_dict_rgb, prefix='backbone_rgb.')
        if self.with_depth and not self.camera_only:
            self.backbone_depth = create_model(self.hparams.backbone_depth['name'], self.hparams.backbone_depth['params'])
            self.has_temporal_depth = self.hparams.backbone_depth['has_temporal']
            if hasattr(self.hparams, 'pretrained_depth_path'):
                print("Loading pretrained Depth backbone from:", self.hparams.pretrained_depth_path)
                state_dict_depth = torch.load(self.hparams.pretrained_depth_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_depth, state_dict_depth, prefix='backbone_depth.')
        if self.with_lidar and not self.camera_only:
            self.backbone_lidar = create_model(self.hparams.backbone_lidar['name'], self.hparams.backbone_lidar['params'])
            self.has_temporal_lidar = self.hparams.backbone_lidar['has_temporal']
            if hasattr(self.hparams, 'pretrained_lidar_path'):
                print("Loading pretrained LIDAR backbone from:", self.hparams.pretrained_lidar_path)
                state_dict_lidar = torch.load(self.hparams.pretrained_lidar_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_lidar, state_dict_lidar, prefix='backbone_lidar.')
        if self.with_mmwave and not self.camera_only:
            self.backbone_mmwave = create_model(self.hparams.backbone_mmwave['name'], self.hparams.backbone_mmwave['params'])
            self.has_temporal_mmwave = self.hparams.backbone_mmwave['has_temporal']
            if hasattr(self.hparams, 'pretrained_mmwave_path'):
                print("Loading pretrained MMWave backbone from:", self.hparams.pretrained_mmwave_path)
                state_dict_mmwave = torch.load(self.hparams.pretrained_mmwave_path, map_location=self.device)['state_dict']
                load_state_dict_part(self.backbone_mmwave, state_dict_mmwave, prefix='backbone_mmwave.')

        if not self.camera_only:
            self.aggregator = create_model(self.hparams.aggregator['name'], self.hparams.aggregator['params'])
        else:
            self.aggregator = None

        if self.with_keypoint_head:
            self.keypoint_head = create_model(self.hparams.keypoint_head['name'], self.hparams.keypoint_head['params'])
        if self.with_smpl_head:
            self.smpl_head = create_model(self.hparams.smpl_head['name'], self.hparams.smpl_head['params'])
        if self.with_camera_head:
            self.camera_head = create_model(self.hparams.camera_head['name'], self.hparams.camera_head['params'])
        
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
            try:
                losses_camera = self.camera_head.loss(feats_agg, batch, pred_dict={})
            except TypeError:
                losses_camera = self.camera_head.loss(feats_agg, batch)
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
                try:
                    losses_camera = self.camera_head.loss(feats_agg, batch, pred_dict=pred_dict)
                except TypeError:
                    losses_camera = self.camera_head.loss(feats_agg, batch)
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
            pred_dict['pred_cameras'] = preds_camera
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
                pred_dict['pred_cameras'] = preds_camera

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
            pred_dict['pred_cameras'] = preds_camera
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
                pred_dict['pred_cameras'] = preds_camera

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
            for i in range(batch_size):
                self.test_preds.append({
                    'sample_id': batch['sample_id'][i],
                    'pred_cameras': torch2numpy(pred_dict['pred_cameras'][i]) if 'pred_cameras' in pred_dict else None,
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
            pred_dict['pred_cameras'] = preds_camera
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
                pred_dict['pred_cameras'] = preds_camera

        return pred_dict

    def _predict_camera(self, feats_agg, batch, pred_dict):
        try:
            return self.camera_head.predict(feats_agg, data_batch=batch, pred_dict=pred_dict)
        except TypeError:
            try:
                return self.camera_head.predict(feats_agg, batch, pred_dict)
            except TypeError:
                try:
                    return self.camera_head.predict(feats_agg, batch)
                except TypeError:
                    return self.camera_head.predict(feats_agg)

    def _attach_keypoint_modalities(self, pred_dict, feats_agg, batch):
        if not self.with_keypoint_head:
            return
        modalities = batch.get("modalities", [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        try:
            outputs = self.keypoint_head.forward(feats_agg, modalities=modalities)
        except TypeError:
            return
        if not isinstance(outputs, dict):
            return
        per_modality = outputs.get("per_modality")
        if per_modality is None:
            return
        for i, modality in enumerate(modalities[: len(per_modality)]):
            mod = modality.lower()
            if mod in {"rgb", "depth"}:
                pred_dict[f"pred_keypoints_2d_{mod}"] = per_modality[i]
            else:
                pred_dict[f"pred_keypoints_3d_{mod}"] = per_modality[i]
                if mod == "lidar":
                    pred_dict["pred_keypoints_pc_centered_input_lidar"] = per_modality[i]

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
            
            # Check and stack each prediction type
            has_cameras = any(item['pred_cameras'] is not None for item in gathered_preds)
            has_pred_keypoints = any(item['pred_keypoints'] is not None for item in gathered_preds)
            has_gt_keypoints = any(item['gt_keypoints'] is not None for item in gathered_preds)
            has_smpl = any('pred_smpl_params' in item for item in gathered_preds)
            
            final_preds['pred_cameras'] = np.stack([item['pred_cameras'] for item in gathered_preds]) if has_cameras else None
            final_preds['pred_keypoints'] = np.stack([item['pred_keypoints'] for item in gathered_preds]) if has_pred_keypoints else None
            final_preds['gt_keypoints'] = np.stack([item['gt_keypoints'] for item in gathered_preds]) if has_gt_keypoints else None
            final_preds['pred_smpl_params'] = np.stack([item['pred_smpl_params'] for item in gathered_preds]) if has_smpl else None
            final_preds['pred_smpl_keypoints'] = np.stack([item['pred_smpl_keypoints'] for item in gathered_preds]) if has_smpl else None
            final_preds['gt_smpl_params'] = np.stack([item['gt_smpl_params'] for item in gathered_preds]) if has_smpl else None

            # sort by sample_ids
            sorted_indices = np.argsort(final_preds['sample_ids'])
            final_preds['sample_ids'] = [final_preds['sample_ids'][i] for i in sorted_indices]
            
            for key in ['pred_cameras', 'pred_keypoints', 'gt_keypoints', 'pred_smpl_params', 'pred_smpl_keypoints', 'gt_smpl_params']:
                if final_preds[key] is not None:
                    final_preds[key] = final_preds[key][sorted_indices]

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
