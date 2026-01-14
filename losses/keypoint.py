import torch.nn as nn

class KeypointLoss(nn.Module):
    def __init__(self, root_joint_idx=-1, loss_type='MSE'):
        super().__init__()
        self.root_joint_idx = root_joint_idx
        match loss_type:
            case 'MSE':
                self.criterion = nn.MSELoss()
            case 'L1':
                self.criterion = nn.L1Loss()
            case _:
                raise ValueError(f'Unsupported loss type: {loss_type}')

    def forward(self, pred_keypoints, gt_keypoints):
        if self.root_joint_idx >= 0:
            pred_keypoints = pred_keypoints - pred_keypoints[:, self.root_joint_idx:self.root_joint_idx+1, :]
            gt_keypoints = gt_keypoints - gt_keypoints[:, self.root_joint_idx:self.root_joint_idx+1, :]
        return self.criterion(pred_keypoints, gt_keypoints)