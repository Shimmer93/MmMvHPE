import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import os
import pickle
from einops import rearrange

from misc.registry import create_model, create_metric, create_optimizer, create_scheduler

class LitModel(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        assert self.with_rgb or self.with_depth or self.with_lidar or self.with_mmwave, "At least one modality should be used."

        if self.with_rgb:
            self.backbone_rgb = create_model(self.hparams.backbone_rgb['name'], self.hparams.backbone_rgb['params'])
            self.has_temporal_rgb = self.hparams.backbone_rgb['has_temporal']
        if self.with_depth:
            self.backbone_depth = create_model(self.hparams.backbone_depth['name'], self.hparams.backbone_depth['params'])
            self.has_temporal_depth = self.hparams.backbone_depth['has_temporal']
        if self.with_lidar:
            self.backbone_lidar = create_model(self.hparams.backbone_lidar['name'], self.hparams.backbone_lidar['params'])
            self.has_temporal_lidar = self.hparams.backbone_lidar['has_temporal']
        if self.with_mmwave:
            self.backbone_mmwave = create_model(self.hparams.backbone_mmwave['name'], self.hparams.backbone_mmwave['params'])
            self.has_temporal_mmwave = self.hparams.backbone_mmwave['has_temporal']

        self.aggregator = create_model(self.hparams.aggregator['name'], self.hparams.aggregator['params'])

        if self.with_camera_head:
            self.camera_head = create_model(self.hparams.camera_head['name'], self.hparams.camera_head['params'])
        if self.with_keypoint_head:
            self.keypoint_head = create_model(self.hparams.keypoint_head['name'], self.hparams.keypoint_head['params'])
        
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)

        self.metrics = {metric['name']: create_metric(metric['name'], metric['params']) for metric in hparams.metrics}

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
        return hasattr(self.hparams, 'keypoint_head')
    
    def forward_rgb(self, frames_rgb):
        """Forward function for RGB frames."""
        if not self.with_rgb or frames_rgb is None:
            return None
        
        B = frames_rgb.shape[0]
        if self.has_temporal_rgb:
            features_rgb = self.backbone_rgb(frames_rgb)
        else:
            frames_rgb = rearrange(frames_rgb, 'b t ... -> (b t) ...')
            features_rgb = self.backbone_rgb(frames_rgb)
            features_rgb = rearrange(features_rgb, '(b t) ... -> b t ...', b=B)

        return features_rgb
    
    def forward_depth(self, frames_depth):
        """Forward function for depth frames."""
        if not self.with_depth or frames_depth is None:
            return None

        B = frames_depth.shape[0]
        if self.has_temporal_depth:
            features_depth = self.backbone_depth(frames_depth)
        else:
            frames_depth = rearrange(frames_depth, 'b t ... -> (b t) ...')
            features_depth = self.backbone_depth(frames_depth)
            features_depth = rearrange(features_depth, '(b t) ... -> b t ...', b=B)

        return features_depth
    
    def forward_lidar(self, frames_lidar):
        """Forward function for LiDAR frames."""
        if not self.with_lidar or frames_lidar is None:
            return None

        B = frames_lidar.shape[0]
        if self.has_temporal_lidar:
            features_lidar = self.backbone_lidar(frames_lidar)
        else:
            frames_lidar = rearrange(frames_lidar, 'b t n c -> (b t) n c')
            features_lidar = self.backbone_lidar(frames_lidar)
            features_lidar = rearrange(features_lidar, '(b t) ... -> b t ...', b=B)

        return features_lidar
    
    def forward_mmwave(self, frames_mmwave):
        """Forward function for mmWave frames."""
        if not self.with_mmwave or frames_mmwave is None:
            return None
        
        B = frames_mmwave.shape[0]
        if self.has_temporal_mmwave:
            features_mmwave = self.backbone_mmwave(frames_mmwave)
        else:
            frames_mmwave = rearrange(frames_mmwave, 'b t n c -> (b t) n c')
            features_mmwave = self.backbone_mmwave(frames_mmwave)
            features_mmwave = rearrange(features_mmwave, '(b t) ... -> b t ...', b=B)

        return features_mmwave

    def extract_features(self, batch):
        feat_rgb = self.forward_rgb(batch.get('input_rgb', None))
        feat_depth = self.forward_depth(batch.get('input_depth', None))
        feat_lidar = self.forward_lidar(batch.get('input_lidar', None))
        feat_mmwave = self.forward_mmwave(batch.get('input_mmwave', None))
        return feat_rgb, feat_depth, feat_lidar, feat_mmwave
    
    def aggregate_features(self, feats, batch):
        return self.aggregator(feats, **batch)

    def training_step(self, batch, batch_idx):
        feats = self.extract_features(batch)
        feats_agg = self.aggregate_features(feats, batch)
        loss_dict = {}
        log_dict = {}
        if self.with_camera_head:
            losses_camera = self.camera_head.loss(feats_agg, batch)
            loss_dict.update(losses_camera)

        if self.with_keypoint_head:
            losses_keypoint = self.keypoint_head.loss(feats_agg, batch)
            loss_dict.update(losses_keypoint)

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
        if self.with_camera_head:
            preds_camera = self.camera_head.predict(feats_agg)
            pred_dict['pred_cameras'] = preds_camera
        if self.with_keypoint_head:
            preds_keypoint = self.keypoint_head.predict(feats_agg)
            pred_dict['pred_keypoints'] = preds_keypoint

        log_dict = {}
        for _, metric in self.metrics.items():
            log_dict[f'val_{metric.name}'] = metric(pred_dict, batch)

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        feats = self.extract_features(batch)
        feats_agg = self.aggregate_features(feats, batch)

        pred_dict = {}
        if self.with_camera_head:
            preds_camera = self.camera_head.predict(feats_agg)
            pred_dict['pred_cameras'] = preds_camera
        if self.with_keypoint_head:
            preds_keypoint = self.keypoint_head.predict(feats_agg)
            pred_dict['pred_keypoints'] = preds_keypoint

        log_dict = {}
        for _, metric in self.metrics.items():
            log_dict[f'test_{metric.name}'] = metric(pred_dict, batch)

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]