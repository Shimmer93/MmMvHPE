import torch
import torch.nn as nn

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        print(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    match loss_type:
        case "l1" | "mae" | "L1" | "MAE":
            # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
            loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
            loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
            loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
        case "l2" | "mse" | "L2" | "MSE":
            # L2 norm for each component
            loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
            loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
            loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL

class CameraLoss(nn.Module):
    def __init__(self, loss_type="l1", weights=[1.0, 1.0, 1.0]):
        super().__init__()
        self.loss_type = loss_type
        self.weights = weights

    def forward(self, pred_camera_enc, gt_camera_enc):
        """
        Computes the camera loss between predicted and ground truth camera encodings.
        
        Args:
            pred_camera_enc: (N, D) predicted camera encodings
            gt_camera_enc: (N, D) ground truth camera encodings

        Returns:
            total_loss: weighted sum of translation, rotation, and focal losses
        """
        loss_T, loss_R, loss_FL = camera_loss_single(
            pred_camera_enc, gt_camera_enc, loss_type=self.loss_type
        )
        losses = {
            "T": loss_T * self.weights[0],
            "R": loss_R * self.weights[1],
            "FL": loss_FL * self.weights[2],
        }
        
        return losses