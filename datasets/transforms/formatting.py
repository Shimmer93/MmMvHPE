import numpy as np
from copy import deepcopy
import torch

class ToTensor():

    def _array_to_tensor(self, data, dtype=torch.float):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        return torch.from_numpy(data).to(dtype)
    
    def _item_to_tensor(self, data, dtype=torch.float):
        return torch.tensor([data], dtype=dtype)
    
    def _list_to_tensor(self, data, dtype=torch.float):
        return torch.from_numpy(np.stack(data, axis=0)).to(dtype)

    @staticmethod
    def _as_tensor_input(data, dtype=torch.float):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        return torch.from_numpy(np.stack(data, axis=0)).to(dtype)

    def __call__(self, sample):
        for key in sample:
            if key.endswith('_affine') or key in ['gt_keypoints', 'gt_smpl_params'] or key.startswith('gt_keypoints'):
                sample[key] = self._array_to_tensor(sample[key])
            elif key.startswith('input_'):
                if isinstance(sample[key], np.ndarray):
                    sample[key] = self._array_to_tensor(sample[key])
                else:
                    sample[key] = self._list_to_tensor(sample[key])
                if key.startswith('input_rgb'):
                    if sample[key].ndim == 4:
                        sample[key] = sample[key].permute(0, 3, 1, 2)  # T H W C -> T C H W
                    elif sample[key].ndim == 5:
                        sample[key] = sample[key].permute(0, 1, 4, 2, 3)  # V T H W C -> V T C H W
                elif key.startswith('input_depth'):
                    # Single-camera grayscale: T H W -> T C H W
                    if sample[key].ndim == 3:
                        sample[key] = sample[key].unsqueeze(1).repeat(1, 3, 1, 1)
                    # Single-camera with channel dim: T H W C -> T C H W
                    elif sample[key].ndim == 4 and sample[key].shape[-1] in (1, 3):
                        sample[key] = sample[key].permute(0, 3, 1, 2)
                    # Multi-camera grayscale: V T H W -> V T C H W
                    elif sample[key].ndim == 4:
                        sample[key] = sample[key].unsqueeze(2).repeat(1, 1, 3, 1, 1)
                    # Multi-camera with channel dim: V T H W C -> V T C H W
                    elif sample[key].ndim == 5 and sample[key].shape[-1] in (1, 3):
                        sample[key] = sample[key].permute(0, 1, 4, 2, 3)
                    
            elif key in ['idx']:
                sample[key] = self._item_to_tensor(sample[key], dtype=torch.long)

        return sample
