import torch
import torch.nn as nn
from models.video_encoders.layers.block import Block


class SimpleAggregator(nn.Module):
    """
    A simple baseline aggregator that fuses multi-modal features by summing them.
    No advanced techniques or tricks - just a straightforward summation.
    """
    
    def __init__(self, input_dims=[512, 512, 512, 512], embed_dim=512, **kwargs):
        """
        Args:
            input_dims: List of input dimensions for each modality [rgb, depth, lidar, mmwave]
                       Can contain None for unused modalities
            embed_dim: Dimension of output features
        """
        super().__init__()
        
        rgb_dim, depth_dim, lidar_dim, mmwave_dim = input_dims
        
        # Only create projection layers for non-None modalities
        self.proj_rgb = nn.Linear(rgb_dim, embed_dim) if rgb_dim is not None else None
        self.proj_depth = nn.Linear(depth_dim, embed_dim) if depth_dim is not None else None
        self.proj_lidar = nn.Linear(lidar_dim, embed_dim) if lidar_dim is not None else None
        self.proj_mmwave = nn.Linear(mmwave_dim, embed_dim) if mmwave_dim is not None else None
        
        # Single transformer block to mix tokens from different modalities
        self.mixer_block = Block(
            dim=embed_dim,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            qk_norm=True,
            init_values=0.01,
        )
    
    def forward(self, features, **kwargs):
        """
        Args:
            features: List of 4 feature tensors from different modalities (can contain None)
                     Each tensor shape: (B, T, N, C) where B=batch, T=time, N=tokens, C=channels
                     T can be different for each modality
        
        Returns:
            List of one tensor with shape (B, M, T, N, C)
        """
        features_rgb, features_depth, features_lidar, features_mmwave = features
        B = 0
        
        # Project each modality
        projected = []
        if features_rgb is not None:
            features_rgb = self.proj_rgb(features_rgb)
            B, T, N, _ = features_rgb.shape
            # Flatten to (B, T*N, C)
            projected.append(features_rgb.reshape(B, T * N, -1))
        if features_depth is not None:
            features_depth = self.proj_depth(features_depth)
            B, T, N, _ = features_depth.shape
            projected.append(features_depth.reshape(B, T * N, -1))
        if features_lidar is not None:
            features_lidar = self.proj_lidar(features_lidar)
            B, T, N, _ = features_lidar.shape
            projected.append(features_lidar.reshape(B, T * N, -1))
        if features_mmwave is not None:
            features_mmwave = self.proj_mmwave(features_mmwave)
            B, T, N, _ = features_mmwave.shape
            projected.append(features_mmwave.reshape(B, T * N, -1))
        
        if len(projected) == 0:
            raise ValueError("No valid features to aggregate")
        
        # Concatenate all tokens from different modalities: (B, total_tokens, C)
        all_tokens = torch.cat(projected, dim=1)
        
        # Apply transformer block to mix all tokens
        mixed_tokens = self.mixer_block(all_tokens)  # (B, total_tokens, C)
        
        # Average over all tokens: (B, C)
        output = mixed_tokens.mean(dim=1)
        
        # Reshape to (B, 1, 1, 1, C) and return as list
        output = output.view(B, 1, 1, 1, -1)
        
        return [output]
