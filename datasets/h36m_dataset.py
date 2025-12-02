import os
import os.path as osp
import cv2
import numpy as np
import yaml
from typing import Callable, List, Optional, Sequence, Tuple, Union

from datasets.base_dataset import BaseDataset

class H36MDataset(BaseDataset):
    def __init__(self, pipeline: List[dict] = []):
        super().__init__(pipeline=pipeline)
        # TODO

    def __getitem__(self, index):
        # TODO

        sample = {}

        # sample should contain:
        # 'input_rgb': List[np.ndarray] of shape (H, W, C)
        # 'input_depth': List[np.ndarray] of shape (H, W)
        # 'input_rgb_camera': ?
        # 'input_depth_camera': ?
        # 'gt_keypoints': np.ndarray of shape (num_joints, 3)
        # 'sample_id': str (just a unique identifier)
        # 'modalities': List[str] (['rgb', 'depth'] here)
        # 'anchor_key': str (if anchor_key is input_rgb, then coordinates are in rgb camera space;
        #                if anchor_key is input_depth, then coordinates are in depth camera space)

        sample = self.pipeline(sample)

        return sample