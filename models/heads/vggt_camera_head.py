import torch
import torch.nn as nn

from .base_head import BaseHead

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.video_encoders.layers import Mlp
from models.video_encoders.layers.block import Block
from .head_act import activate_pose
from misc.pose_enc import pose_encoding_to_extri_intri


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        # print("Camera head output shape:", pred_pose_enc_list[-1].shape)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift

class VGGTCameraHead(BaseHead):
    def __init__(
            self,
            losses,
            dim_in: int = 2048,
            trunk_depth: int = 4,
            pose_encoding_type: str = "absT_quaR_FoV",
            num_heads: int = 16,
            mlp_ratio: int = 4,
            init_values: float = 0.01,
            trans_act: str = "linear",
            quat_act: str = "linear",
            fl_act: str = "relu",
            last_n_layers=-1,
    ):
        super().__init__(losses)
        self.camera_head = CameraHead(
            dim_in=dim_in,
            trunk_depth=trunk_depth,
            pose_encoding_type=pose_encoding_type,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            trans_act=trans_act,
            quat_act=quat_act,
            fl_act=fl_act,
        )
        self.last_n_layers = last_n_layers

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        x = aggregated_tokens_list
        if self.last_n_layers > 0:
            x = x[-self.last_n_layers:]
        x = torch.concatenate(x, dim=-1)

        # x.shape: B, T, num_camera_tokens + num_smpl_tokens + num_joints, C
        B, T, N, C = x.shape
        num_camera_tokens = N - 1 - 24
        x = x[:, -1, :num_camera_tokens, :]
        x = x.unsqueeze(-2)  

        last_output = x
        # print("Camera head input shape:", last_output.shape)
        # last_output = aggregated_tokens_list[-1]
        # B, M, T, N, C = last_output.shape
        # last_output = last_output.mean(dim=2) # Average over temporal dimension
        return self.camera_head([last_output], num_iterations)
    
    def loss(self, x, data_batch):
        pred_camera_enc_list = self.forward(x, num_iterations=data_batch.get('num_camera_iterations',4))
        pred_camera_encs = pred_camera_enc_list[-1]
        pred_camera_enc_list = [pred_camera_encs[:,i,...] for i in range(pred_camera_encs.shape[1])]
        modalities = data_batch['modalities']
        # print("Modalities for camera loss:", modalities)
        assert pred_camera_encs.shape[1] == len(modalities[0]), "Number of predicted camera encodings must match number of modalities."

        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            for pred_camera_enc, modality in zip(pred_camera_enc_list, modalities[0]):
                loss_output = loss_fn(pred_camera_enc, data_batch[f'gt_camera_{modality}'])
                if isinstance(loss_output, dict):
                    for k, v in loss_output.items():
                        losses[f"{loss_name}_{modality}_{k}"] = (v, loss_weight)
                else:
                    losses[f"{loss_name}_{modality}"] = (loss_output, loss_weight)
        return losses
    
    def predict(self, x, num_iterations: int = 4):
        return self.forward(x, num_iterations)