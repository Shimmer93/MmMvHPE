import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

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
    def forward(self, feature_list, lidar_points, modality_list):
        # example:
        # feature_list = [rgb_feature, mmwave_feature, lidar_feature]
        # modality_list = [True, False, True, True]
        feature_flag = 0
        for i in range(len(modality_list)):
            if modality_list[i] == True:
                if i == 0:
                    rgb_feature = feature_list[feature_flag]
                elif i == 1:
                    depth_feature = feature_list[feature_flag]
                elif i == 2:
                    mmwave_feature = feature_list[feature_flag]
                elif i == 3:
                    lidar_feature = feature_list[feature_flag]
                feature_flag += 1
            else:
                continue
        if sum (modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            projected_feature_list = []
            if modality_list[0] == True:
                projected_feature_list.append(self.rgb_linear_projection(rgb_feature.permute(0, 2, 1)))
            if modality_list[1] == True:
                projected_feature_list.append(self.depth_linear_projection(depth_feature.permute(0, 2, 1)))
            if modality_list[2] == True:
                projected_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
            if modality_list[3] == True:
                projected_feature_list.append(self.lidar_linear_projection(lidar_feature.permute(0, 2, 1)))
            projected_feature = torch.cat(projected_feature_list, dim=2).permute(0, 2, 1)
            "projected_feature shape: B, 32*n, 512"
            if modality_list[3] == True:
                feature_shape = projected_feature.shape
                new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
                'new_xyz shape: B, 32, 3'
                pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
                'pos_enc shape: B, 32, 512'
                projected_feature += pos_enc
            else:
                pass
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
    def __init__(self, input_dim, output_dim, num_modalities=4, dim_expansion=2, hidden_dim=1024, num_heads=8, dim_heads=64, model_depth=2):
        super(XFiAggregator, self).__init__()
        self.input_dim = input_dim
        self.linear_projector = linear_projector(input_dim, output_dim)
        self.x_fusion = X_Fusion(num_modalities, output_dim, dim_expansion, hidden_dim, num_feature=32, num_heads=num_heads, dim_heads=dim_heads, model_depth=model_depth)
    
    def forward(self, features, **kwargs):
        # features_rgb: B T C H W
        # features_depth: B T C H W
        # features_mmwave: B T N_mm C_mm
        # features_lidar: B T N_ld C_ld

        features_rgb, features_depth, features_lidar, features_mmwave = features

        feature_list = []
        modality_list = []
        if features_rgb is not None:
            b, t, c, h, w = features_rgb.shape
            features_rgb = features_rgb.reshape(b * t, self.input_dim, c // self.input_dim, h, w)
            features_rgb = rearrange(features_rgb, 'bt d c h w -> bt (c h w) d')
            feature_list.append(features_rgb)
            modality_list.append(True)
        else:
            feature_list.append(None)
            modality_list.append(False)
        if features_depth is not None:
            b, t, c, h, w = features_depth.shape
            features_depth = features_depth.reshape(b * t, self.input_dim, c // self.input_dim, h, w)
            features_depth = rearrange(features_depth, 'bt d c h w -> bt (c h w) d')
            feature_list.append(features_depth)
            modality_list.append(True)
        else:
            feature_list.append(None)
            modality_list.append(False)
        if features_lidar is not None:
            features_lidar = rearrange(features_lidar, 'b t n c -> (b t) n c')
            feature_list.append(features_lidar)
            modality_list.append(True)
        else:
            feature_list.append(None)
            modality_list.append(False)
        if features_mmwave is not None:
            features_mmwave = rearrange(features_mmwave, 'b t n c -> (b t) n c')
            feature_list.append(features_mmwave)
            modality_list.append(True)
        else:
            feature_list.append(None)
            modality_list.append(False)
        lidar_points = kwargs.get('input_lidar', None)
        lidar_points = rearrange(lidar_points, 'b t n c -> (b t) n c')
        projected_feature = self.linear_projector(feature_list, lidar_points, modality_list)
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
    fused_feature = aggregator((rgb_feature, depth_feature, mmwave_feature, lidar_feature), input_lidar=input_lidar)
    print(f"Fused feature shape: {fused_feature.shape}")