import numpy as np
import torch

from smplpytorch.native.webuser.serialization import ready_arguments
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

class SMPL(SMPL_Layer):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self,
                 center_idx=None,
                 gender='neutral',
                 model_path='/opt/data/SMPL_NEUTRAL.pkl'):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_path: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        torch.nn.Module.__init__(self)

        self.center_idx = center_idx
        self.gender = gender

        self.model_path = model_path
        print(f"[DEBUG]: Loading SMPL model from {self.model_path}...")

        smpl_data = ready_arguments(self.model_path)
        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24