import numpy as np
from copy import deepcopy
import torch

class ToTensor():

    def _array_to_tensor(self, data, dtype=torch.float):
        return torch.from_numpy(data).to(dtype)
    
    def _item_to_tensor(self, data, dtype=torch.float):
        return torch.tensor([data], dtype=dtype)
    
    def _list_to_tensor(self, data, dtype=torch.float):
        return torch.from_numpy(np.stack(data, axis=0)).to(dtype)

    def __call__(self, sample):
        for key in sample:
            if key.endswith('_affine') or key in ['gt_keypoints']:
                sample[key] = self._array_to_tensor(sample[key])
            elif key.startswith('input_'):
                sample[key] = self._list_to_tensor(sample[key])
                if key.startswith('input_rgb') or key.startswith('input_depth'):
                    sample[key] = sample[key].permute(0, 3, 1, 2)  # T H W C -> T C H W
                    
            elif key in ['idx']:
                sample[key] = self._item_to_tensor(sample[key], dtype=torch.long)

        return sample