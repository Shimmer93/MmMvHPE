import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import warnings

CANONICAL_MODALITIES = ("rgb", "depth", "mmwave", "lidar")

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def selective_pos_enc(xyz, npoint):
    """
    Input:
        xyz: input points position data, [B, N, 3]
    Return:
        new_xyz: sampled points position data, [B, S, 3]
        out: new features of sampled points, [B, S, C]
    """
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    return new_xyz

class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_projector, self).__init__()
        '''Conv 1d layer for each modality'''
        self.rgb_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.depth_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.mmwave_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.lidar_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.pos_enc_layer = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    def forward(self, feature_map, lidar_points, active_modalities):
        if not active_modalities:
            raise ValueError("At least one modality should be selected.")

        projected_feature_list = []
        for modality in active_modalities:
            feature = feature_map.get(modality, None)
            if feature is None:
                raise RuntimeError(
                    f"Missing prepared feature for modality '{modality}' during projection."
                )
            if modality == "rgb":
                projected_feature_list.append(self.rgb_linear_projection(feature.permute(0, 2, 1)))
            elif modality == "depth":
                projected_feature_list.append(self.depth_linear_projection(feature.permute(0, 2, 1)))
            elif modality == "mmwave":
                projected_feature_list.append(self.mmwave_linear_projection(feature.permute(0, 2, 1)))
            elif modality == "lidar":
                projected_feature_list.append(self.lidar_linear_projection(feature.permute(0, 2, 1)))
            else:
                raise ValueError(f"Unsupported modality '{modality}'.")

        projected_feature = torch.cat(projected_feature_list, dim=2).permute(0, 2, 1)
        # projected_feature shape: B, 32*n, 512
        if "lidar" in active_modalities:
            if lidar_points is None:
                raise RuntimeError(
                    "LiDAR positional encoding requires `input_lidar`, but it is missing."
                )
            if lidar_points.dim() != 3 or lidar_points.shape[-1] != 3:
                raise RuntimeError(
                    f"`input_lidar` for positional encoding must be shape [BT, N, 3], got {tuple(lidar_points.shape)}."
                )
            feature_shape = projected_feature.shape
            new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
            # new_xyz shape: B, 32*n, 3
            pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
            # pos_enc shape: B, 32*n, 512
            projected_feature += pos_enc
        return projected_feature
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 512, num_heads = 8, dropout = 0.0):
        super(MultiHeadAttention,self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        out = self.pool(out)
        return out

class qkv_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(qkv_Attention,self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qkv):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class kv_projection(nn.Module):
    def __init__(self, dim, expension = 2):
        super(kv_projection,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim * expension),
            nn.LayerNorm(dim * expension),
            nn.ReLU(),
            nn.Linear(dim * expension, dim)
        )
        # self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
    
    def forward(self, x):
        x = self.MLP(x)
        # q = self.to_q(self.norm(x))
        k = self.to_k(self.norm(x))
        v = self.to_v(self.norm(x))
        kv = [k, v]
        return kv


class cross_modal_transformer(nn.Module):
    def __init__(self, num_feature=32, max_num_modality=5, dim_expansion=2, emb_size = 512, num_heads = 8, dropout=0.):
        super(cross_modal_transformer,self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.ffw = FeedForward(emb_size, emb_size*dim_expansion, dropout)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, feature_embedding, modality_list):
        feature_embedding_ = self.attention(feature_embedding) + self.pool(feature_embedding)
        out_feature_embedding = self.ffw(feature_embedding_) + feature_embedding_
        return out_feature_embedding

class fusion_transformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads):
        super(fusion_transformer,self).__init__()
        self.mutihead_attention = qkv_Attention(dim, num_heads, dim_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
    
    def forward(self, feature_embedding, kv):
        qkv = (feature_embedding, kv[0], kv[1])
        x = self.mutihead_attention(qkv) + feature_embedding
        new_feature_embedding = self.feed_forward(x) + x
        return new_feature_embedding


class fusion_transformer_block(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, num_modality):
        super(fusion_transformer_block,self).__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_modality):
            self.transformer_layers.append(fusion_transformer(dim, hidden_dim, num_heads, dim_heads))

    def forward(self, feature_embedding, kv_list, modality_list):
        "each modality cross attention fusion"
        transformer_layer_idx = 0
        feature_idx_ = 0
        features_list = []
        for layer in self.transformer_layers:
            if modality_list[transformer_layer_idx] == True:
                new_feature_embedding = layer(feature_embedding, kv_list[feature_idx_])
                features_list.append(new_feature_embedding)
                transformer_layer_idx += 1
                feature_idx_ += 1
            else:
                transformer_layer_idx += 1
        feature_embedding = torch.cat(features_list, dim=1)
        return feature_embedding

class X_Fusion(nn.Module):
    def __init__(self, num_modalities, dim, qkv_hidden_expansion, hidden_dim, num_feature, num_heads, dim_heads, model_depth):
        super(X_Fusion,self).__init__()
        self.kv_layers = nn.ModuleList([])
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_modalities):
            self.kv_layers.append(kv_projection(dim, qkv_hidden_expansion))
        self.cross_attention_transformers = fusion_transformer_block(dim, hidden_dim, num_heads, dim_heads, num_modalities)
        self.cross_modal_transformer = cross_modal_transformer(num_feature, num_modalities, qkv_hidden_expansion, dim, num_heads, dropout=0.)
        self.depth = model_depth

    def forward(self, feature, modality_list):
        num_modalities = sum(modality_list)
        feature_list = list(feature.chunk(num_modalities, dim = 1))
        # generate kv_list for all modalities 
        kv_list = []
        kv_layer_idx = 0
        feature_idx = 0
        for kv_layer in self.kv_layers:
            if modality_list[kv_layer_idx] == True:
                kv = kv_layer(feature_list[feature_idx])
                kv_list.append(kv)
                feature_idx += 1
                kv_layer_idx += 1
            else:
                kv_layer_idx += 1
                continue
        # output kv_list: [[k_1, v_1], [k_2, v_2], ...]
        # first iter on cross_modal_transformer
        feature_embedding = self.cross_modal_transformer(feature, modality_list)

        for _ in range(self.depth):
            # feature_embedding shape: B, 32, 512
            feature_embedding = self.cross_attention_transformers(feature_embedding, kv_list, modality_list)
            # feature_embedding shape: B, 32*n, 512
            feature_embedding = self.cross_modal_transformer(feature_embedding, modality_list)

        return feature_embedding

class XFiAggregator(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_modalities=4,
        dim_expansion=2,
        hidden_dim=1024,
        num_heads=8,
        dim_heads=64,
        model_depth=2,
        active_modalities=None,
    ):
        super(XFiAggregator, self).__init__()
        if int(num_modalities) != len(CANONICAL_MODALITIES):
            raise ValueError(
                f"XFiAggregator expects num_modalities={len(CANONICAL_MODALITIES)} "
                f"for canonical mapping {CANONICAL_MODALITIES}, got {num_modalities}."
            )
        self.input_dim = input_dim
        self.canonical_modalities = tuple(CANONICAL_MODALITIES)
        self._warned_implicit_modalities = False
        self.active_modalities = self._validate_and_canonicalize_active_modalities(active_modalities)
        self.linear_projector = linear_projector(input_dim, output_dim)
        self.x_fusion = X_Fusion(
            len(self.canonical_modalities),
            output_dim,
            dim_expansion,
            hidden_dim,
            num_feature=32,
            num_heads=num_heads,
            dim_heads=dim_heads,
            model_depth=model_depth,
        )

    def _validate_and_canonicalize_active_modalities(self, active_modalities):
        if active_modalities is None:
            return None
        if not isinstance(active_modalities, (list, tuple)):
            raise ValueError(
                "`active_modalities` must be a list/tuple of modality names, "
                f"got {type(active_modalities).__name__}."
            )
        requested = [str(x).strip().lower() for x in active_modalities]
        if len(requested) == 0:
            raise ValueError("`active_modalities` cannot be empty.")
        duplicates = [m for m in set(requested) if requested.count(m) > 1]
        if duplicates:
            raise ValueError(
                f"`active_modalities` contains duplicates: {sorted(duplicates)}."
            )
        unknown = [m for m in requested if m not in self.canonical_modalities]
        if unknown:
            raise ValueError(
                f"Unsupported modalities in `active_modalities`: {sorted(unknown)}. "
                f"Supported: {list(self.canonical_modalities)}."
            )
        # Always convert to canonical order internally.
        requested_set = set(requested)
        return [m for m in self.canonical_modalities if m in requested_set]

    def _resolve_active_modalities(self, feature_map):
        if self.active_modalities is not None:
            return list(self.active_modalities)
        inferred = [m for m in self.canonical_modalities if feature_map.get(m, None) is not None]
        if not inferred:
            raise RuntimeError(
                "Failed to infer active modalities because all modality features are None. "
                "Set `aggregator.params.active_modalities` explicitly."
            )
        if not self._warned_implicit_modalities:
            warnings.warn(
                "XFiAggregator inferred active_modalities from non-None features. "
                "Please set `aggregator.params.active_modalities` explicitly in config."
            )
            self._warned_implicit_modalities = True
        return inferred

    def _prepare_image_feature(self, feature, modality):
        if feature is None:
            return None
        if feature.dim() != 5:
            raise RuntimeError(
                f"{modality} feature must be 5D [B, T, C, H, W], got {tuple(feature.shape)}."
            )
        b, t, c, h, w = feature.shape
        if c % self.input_dim != 0:
            raise RuntimeError(
                f"{modality} feature channel dimension ({c}) must be divisible by input_dim ({self.input_dim})."
            )
        feature = feature.reshape(b * t, self.input_dim, c // self.input_dim, h, w)
        return rearrange(feature, "bt d c h w -> bt (c h w) d")

    @staticmethod
    def _prepare_point_feature(feature, modality):
        if feature is None:
            return None
        if feature.dim() != 4:
            raise RuntimeError(
                f"{modality} feature must be 4D [B, T, N, C], got {tuple(feature.shape)}."
            )
        return rearrange(feature, "b t n c -> (b t) n c")

    @staticmethod
    def _prepare_lidar_points(lidar_points):
        if lidar_points is None:
            return None
        if lidar_points.dim() != 4 or lidar_points.shape[-1] != 3:
            raise RuntimeError(
                f"`input_lidar` must be shape [B, T, N, 3], got {tuple(lidar_points.shape)}."
            )
        return rearrange(lidar_points, "b t n c -> (b t) n c")
    
    def forward(self, features, **kwargs):
        # features_rgb: B T C H W
        # features_depth: B T C H W
        # features_mmwave: B T N_mm C_mm
        # features_lidar: B T N_ld C_ld
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError(
                "XFiAggregator expects `features` as (rgb, depth, lidar, mmwave)."
            )

        # model_api extract_features returns: (rgb, depth, lidar, mmwave)
        features_rgb, features_depth, features_lidar, features_mmwave = features

        feature_map = {
            "rgb": self._prepare_image_feature(features_rgb, "rgb"),
            "depth": self._prepare_image_feature(features_depth, "depth"),
            "mmwave": self._prepare_point_feature(features_mmwave, "mmwave"),
            "lidar": self._prepare_point_feature(features_lidar, "lidar"),
        }
        active_modalities = self._resolve_active_modalities(feature_map)
        missing_modalities = [m for m in active_modalities if feature_map[m] is None]
        if missing_modalities:
            present = [m for m in self.canonical_modalities if feature_map[m] is not None]
            raise RuntimeError(
                "Configured active modalities are missing features at aggregation time. "
                f"Missing={missing_modalities}, Present={present}, Active={active_modalities}."
            )

        modality_list = [m in active_modalities for m in self.canonical_modalities]
        lidar_points = self._prepare_lidar_points(kwargs.get("input_lidar", None))
        if "lidar" in active_modalities and lidar_points is None:
            raise RuntimeError(
                "Configured active modalities include 'lidar' but batch is missing `input_lidar`."
            )
        projected_feature = self.linear_projector(feature_map, lidar_points, active_modalities)
        fused_feature = self.x_fusion(projected_feature, modality_list)
        return fused_feature
    

if __name__ == '__main__':
    # test code
    batch_size = 2
    time_steps = 4
    rgb_feature = torch.randn(batch_size, time_steps, 2048, 7, 7)
    depth_feature = torch.randn(batch_size, time_steps, 512, 4, 2)
    lidar_feature = torch.randn(batch_size, time_steps, 64, 512)
    mmwave_feature = torch.randn(batch_size, time_steps, 32, 512)
    input_lidar = torch.randn(batch_size, time_steps, 1024, 3)

    aggregator = XFiAggregator(input_dim=512, output_dim=512, num_modalities=4, dim_expansion=2, hidden_dim=512, num_heads=8, dim_heads=64, model_depth=4)
    fused_feature = aggregator((rgb_feature, depth_feature, lidar_feature, mmwave_feature), input_lidar=input_lidar)
    print(f"Fused feature shape: {fused_feature.shape}")
