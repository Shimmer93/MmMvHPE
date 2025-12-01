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
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
            else:
                batch_data[key] = [sample[key] for sample in batch]
        return batch_data