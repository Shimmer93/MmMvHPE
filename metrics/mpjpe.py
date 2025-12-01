import numpy as np

from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.utils.tensor_utils import to_numpy

class MPJPE:
    def __init__(self, affix=None, align='none'):
        self.affix = affix
        self.align = align
        if align == 'procrustes':
            self.name = f'pampjpe_{affix}' if affix is not None else 'pampjpe'
        elif align == 'scale':
            self.name = f'scale_mpjpe_{affix}' if affix is not None else 'scale_mpjpe'
        else:
            self.name = f'mpjpe_{affix}' if affix is not None else 'mpjpe'

    def __call__(self, preds, targets):
        if self.affix is not None:
            pred_keypoints = to_numpy(preds[f'pred_keypoints_{self.affix}'])
            target_keypoints = to_numpy(targets[f'gt_keypoints_{self.affix}'])
        else:
            pred_keypoints = to_numpy(preds['pred_keypoints'])
            target_keypoints = to_numpy(targets['gt_keypoints'])

        mask = np.ones_like(target_keypoints[..., 0], dtype=bool)
        mpjpe = keypoint_mpjpe(pred_keypoints, target_keypoints, mask, alignment = 'none')
        return mpjpe