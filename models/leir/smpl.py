import pickle
import numpy as np
import torch
import torch.nn as nn

from misc.rotation import batch_rodrigues


class LEIRSMPL(nn.Module):
    def __init__(self, model_file: str):
        super().__init__()
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')

        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]

        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense())
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer('posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int64)))

        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer(
            'parent',
            torch.LongTensor([id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])])
        )

        self.pose_shape = [24, 3]
        self.beta_shape = [10]

        self.requires_grad_(False)

    def forward(self, pose: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        device = pose.device
        batch_size = pose.shape[0]

        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template

        J = torch.stack([torch.matmul(self.J_regressor, v_shaped[i]) for i in range(batch_size)], dim=0)

        if pose.ndimension() == 4:
            R = pose
        elif pose.ndimension() == 2:
            pose_cube = pose.reshape(-1, 3)
            R = batch_rodrigues(pose_cube).view(batch_size, 24, 3, 3)
        else:
            raise ValueError(f"Unsupported pose shape: {pose.shape}")

        I_cube = torch.eye(3, device=device)[None, None, :]

        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)

        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]

        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1, device=device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3, device=device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)

        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4, 4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]

        return v

    def get_full_joints(self, vertices: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
