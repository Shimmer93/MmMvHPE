import torch
from torch.utils.data import Dataset
from typing import List

from misc.registry import create_tranform
from datasets.transforms import RandomApply, ComposeTransform

class BaseDataset(Dataset):
    def __init__(self, pipeline: List[dict] = []):
        super().__init__()
        self.pipeline = self._load_pipeline(pipeline)

    def _load_pipeline(self, pipeline_cfg):        
        tsfms = []
        for tsfm in pipeline_cfg:
            tsfm_class = create_tranform(tsfm['name'], tsfm.get('params', {}))
            if 'prob' in tsfm:
                tsfm_class = RandomApply([tsfm_class], prob=tsfm['prob'])
            tsfms.append(tsfm_class)
        return ComposeTransform(tsfms)
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        for key in all_keys:
            values = []
            for sample in batch:
                if key in sample:
                    values.append(sample[key])
                elif key.endswith('_affine'):
                    values.append(torch.eye(4, dtype=torch.float32))
                else:
                    values.append(None)

            if all(isinstance(v, torch.Tensor) for v in values if v is not None):
                if any(v is None for v in values):
                    batch_data[key] = values
                else:
                    batch_data[key] = torch.stack(values, dim=0)
            else:
                batch_data[key] = values
        return batch_data
