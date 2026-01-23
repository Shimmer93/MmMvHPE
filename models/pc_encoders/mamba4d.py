import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pc_encoders'))
print("[DEBUG]: ROOT_DIR:", ROOT_DIR)

from .modules.intra_mamba import *
from .modules.mamba import *
from .modules.utils_mamba import Group, Encoder
from .modules.point_4d_convolution import *

class MAMBA4DEncoder(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, mlp_dim, num_classes,                                                 # output
                 depth_mamba_inter, rms_norm, 
                 drop_out_in_block, drop_path,
                 depth_mamba_intra, intra, mode = 'all'                                  # intra-mamba or p4dconv
                 ):
        super().__init__()

        feature_extraction_params = {
            'in_planes': 0,
            'mlp_planes': [dim],
            'mlp_batch_norm': [False],
            'mlp_activation': [False],
            'spatial_kernel_size': [radius, nsamples],
            'spatial_stride': spatial_stride,
            'temporal_kernel_size': temporal_kernel_size,
            'temporal_stride': temporal_stride,
            'temporal_padding': [1, 1],   # Changed from [1, 0] to [1, 1] to prevent empty list on short clips
            'operator': '+',
            'spatial_pooling': 'max',
            'temporal_pooling': 'max',
            'depth_mamba_intra': depth_mamba_intra
        }
        
        self.tube_embedding = IntraMamba(**feature_extraction_params) if intra else P4DConv(**feature_extraction_params)
        self.mode = mode 

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.mambaBlocks = MixerModel(d_model=dim,
                            n_layer=depth_mamba_inter,
                            rms_norm=rms_norm,
                            drop_out_in_block=drop_out_in_block,
                            drop_path=drop_path)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_dim, num_classes),
        # )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()

        if self.mode == 'only_h':
            input_feat = input[:,:,:,2:3].clone()
        elif self.mode == 'xyz':
            input_feat = input[:,:,:,:3].clone()
        elif self.mode == 'd':
            input_feat = input[:,:,:,3:4].clone()
        elif self.mode == 'all':
            input_feat = input[:,:,:,3:5].clone()
        xyzs, features = self.tube_embedding(input[:,:,:,:3], input_feat.permute(0,1,3,2))
        print("[DEBUG]: This is the line 77 of mamba4d.py")
        print("[DEBUG]: The xyzs shape is, ", xyzs.shape)
        print("[DEBUG]: The features shape is, ", features.shape)
        B, L, C, n = features.shape

        xyzts = []
        # sort
        x_labels = []
        y_labels = []
        z_labels = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
            # sort
            xyz_index = torch.argsort(xyzt, dim=1)
            x_labels.append(xyz_index[:, :, 0].unsqueeze(-1))
            y_labels.append(xyz_index[:, :, 1].unsqueeze(-1))
            z_labels.append(xyz_index[:, :, 2].unsqueeze(-1))

        # sort
        x_labels, y_labels, z_labels = torch.cat(x_labels, dim=1), torch.cat(y_labels, dim=1), torch.cat(z_labels, dim=1)
        x_labels = torch.argsort(x_labels, dim=1, stable=True)
        y_labels = torch.argsort(y_labels, dim=1, stable=True)
        z_labels = torch.argsort(z_labels, dim=1, stable=True)

        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        # Fix features reshape: [B, L, C, n] -> [B, L, n, C] -> [B, L*n, C]
        # features = features.permute(0, 1, 3, 2).contiguous()
        features = features.reshape(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]) # [B, L*n, C]
        print("[DEBUG]: The features reshaped shape is, ", features.shape)
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        print("[DEBUG]: The pos_embedding output shape is, ", xyzts.shape)

        embedding = xyzts + features

        # sort
        embedding_x = torch.gather(embedding, 1, x_labels.expand(-1, -1, embedding.size(-1)))
        embedding_y = torch.gather(embedding, 1, y_labels.expand(-1, -1, embedding.size(-1)))
        embedding_z = torch.gather(embedding, 1, z_labels.expand(-1, -1, embedding.size(-1)))

        embedding = torch.cat([embedding_z, embedding_x, embedding_y], dim=1)

        if self.emb_relu:
            embedding = self.emb_relu(embedding)
            
        print("[DEBUG]: The embedding shape is, ", embedding.shape)

        output = self.mambaBlocks(embedding)
        
        # # [FIX START] Properly aggregate the 3 views (Z, X, Y) back to original point order
        # # 1. Split the output back into the 3 sorted streams
        # # Output shape is [B, 3*L*n, C]. View it as [B, 3, L*n, C]
        # output = output.view(B, 3, L*n, -1)
        # out_z = output[:, 0]
        # out_x = output[:, 1]
        # out_y = output[:, 2]

        # # 2. Reverse the sorting (Scatter back to original order)
        # # We accumulate features into a single tensor initialized to zero.
        # # This effectively sums the processed features from the 3 views for each specific point.
        # combined_features = torch.zeros_like(out_z)
        
        # # Expand indices to match feature dimension C
        # C_out = output.shape[-1]
        # idx_z = z_labels.expand(-1, -1, C_out)
        # idx_x = x_labels.expand(-1, -1, C_out)
        # idx_y = y_labels.expand(-1, -1, C_out)
        
        # # scatter_add_(dim, index, src): adds src elements into self at indices
        # combined_features.scatter_add_(1, idx_z, out_z)
        # combined_features.scatter_add_(1, idx_x, out_x)
        # combined_features.scatter_add_(1, idx_y, out_y)

        # # 3. Reshape back to [B, L, N, C]
        output = output.reshape(B, L, -1, output.shape[-1])  # B L N C
        print("[DEBUG]: The shape of the output is: ", output.shape)
        # [FIX END]
        print("[DEBUG]: MAMBA4DEncoder forward pass completed.")
        print(f"[DEBUG]: Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[DEBUG]: Current memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        return output
