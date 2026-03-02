"""
SMPL-specific metrics for evaluating SMPL parameter predictions.

These metrics understand SMPL parameter format and handle conversion to
keypoints/vertices using the SMPL model for evaluation.
"""

import numpy as np
import torch
from misc.utils import torch2numpy as to_numpy
from .mpjpe import mpjpe_func, pampjpe_func, pcmpjpe_func


def smpl_to_keypoints(smpl_params, smpl_model, device='cuda'):
    """
    Convert SMPL parameters to 3D keypoints.
    
    Args:
        smpl_params: Dictionary with SMPL parameters (can be torch tensors or numpy arrays)
        smpl_model: SMPL model instance
        device: Device to run SMPL model on
    
    Returns:
        keypoints: (B, 24, 3) numpy array of 3D joint positions
    """
    # Convert to tensors if needed
    if isinstance(smpl_params['global_orient'], np.ndarray):
        global_orient = torch.from_numpy(smpl_params['global_orient']).float().to(device)
        body_pose = torch.from_numpy(smpl_params['body_pose']).float().to(device)
        betas = torch.from_numpy(smpl_params['betas']).float().to(device)
        transl = torch.from_numpy(smpl_params['transl']).float().to(device)
    else:
        global_orient = smpl_params['global_orient'].to(device)
        body_pose = smpl_params['body_pose'].to(device)
        betas = smpl_params['betas'].to(device)
        transl = smpl_params['transl'].to(device)
    
    # Ensure batch dimension
    if global_orient.dim() == 1:
        global_orient = global_orient.unsqueeze(0)
        body_pose = body_pose.unsqueeze(0)
        betas = betas.unsqueeze(0)
        transl = transl.unsqueeze(0)
    
    B = global_orient.shape[0]
    
    # Concatenate pose parameters
    pose = torch.cat([global_orient, body_pose], dim=1)  # (B, 72)
    
    # Pad betas if necessary
    expected_betas = smpl_model.th_betas.shape[1]
    if betas.shape[1] < expected_betas:
        betas_padded = torch.zeros(B, expected_betas, device=device, dtype=betas.dtype)
        betas_padded[:, :betas.shape[1]] = betas
        betas = betas_padded
    
    # Get keypoints from SMPL model
    with torch.no_grad():
        output = smpl_model(pose, betas, transl)
        # Output is typically (vertices, joints)
        if isinstance(output, (list, tuple)):
            joints = output[1]  # (B, 24, 3)
        else:
            joints = output
    
    return joints.cpu().numpy()


def smpl_to_vertices(smpl_params, smpl_model, device='cuda'):
    """
    Convert SMPL parameters to mesh vertices.
    
    Args:
        smpl_params: Dictionary with SMPL parameters
        smpl_model: SMPL model instance
        device: Device to run SMPL model on
    
    Returns:
        vertices: (B, 6890, 3) numpy array of mesh vertices
    """
    # Convert to tensors if needed
    if isinstance(smpl_params['global_orient'], np.ndarray):
        global_orient = torch.from_numpy(smpl_params['global_orient']).float().to(device)
        body_pose = torch.from_numpy(smpl_params['body_pose']).float().to(device)
        betas = torch.from_numpy(smpl_params['betas']).float().to(device)
        transl = torch.from_numpy(smpl_params['transl']).float().to(device)
    else:
        global_orient = smpl_params['global_orient'].to(device)
        body_pose = smpl_params['body_pose'].to(device)
        betas = smpl_params['betas'].to(device)
        transl = smpl_params['transl'].to(device)
    
    # Ensure batch dimension
    if global_orient.dim() == 1:
        global_orient = global_orient.unsqueeze(0)
        body_pose = body_pose.unsqueeze(0)
        betas = betas.unsqueeze(0)
        transl = transl.unsqueeze(0)
    
    B = global_orient.shape[0]
    
    # Concatenate pose parameters
    pose = torch.cat([global_orient, body_pose], dim=1)
    
    # Pad betas if necessary
    expected_betas = smpl_model.th_betas.shape[1]
    if betas.shape[1] < expected_betas:
        betas_padded = torch.zeros(B, expected_betas, device=device, dtype=betas.dtype)
        betas_padded[:, :betas.shape[1]] = betas
        betas = betas_padded
    
    # Get vertices from SMPL model
    with torch.no_grad():
        output = smpl_model(pose, betas, transl)
        if isinstance(output, (list, tuple)):
            vertices = output[0]  # (B, 6890, 3)
        else:
            vertices = output
    
    return vertices.cpu().numpy()


class SMPL_MPJPE:
    """
    Mean Per Joint Position Error for SMPL predictions.
    
    Converts SMPL parameters to keypoints using SMPL model, then evaluates.
    Expects predictions dict with 'pred_smpl' key from SMPLHead.predict().
    """
    
    def __init__(self, smpl_model_path='weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', affix=None):
        """
        Args:
            smpl_model_path: Path to SMPL model file (.pkl)
            affix: Optional suffix for metric name
        """
        self.smpl_model_path = smpl_model_path
        self.affix = affix
        self.name = f'smpl_mpjpe_{affix}' if affix is not None else 'smpl_mpjpe'
        self._smpl_model = None
    
    @property
    def smpl_model(self):
        """Lazy initialization of SMPL model."""
        if self._smpl_model is None:
            from models.smpl import SMPL
            self._smpl_model = SMPL(model_path=self.smpl_model_path)
            self._smpl_model = self._smpl_model.cuda()
        return self._smpl_model

    def __call__(self, preds, targets):
        # Extract SMPL parameters from prediction
        pred_smpl = preds.get('pred_smpl', None)
        if pred_smpl is None:
            raise ValueError("'pred_smpl' not found in predictions. Make sure you're using SMPLHead.")
        
        # Convert SMPL parameters to keypoints
        pred_keypoints = smpl_to_keypoints(pred_smpl, self.smpl_model, device='cuda')
        
        # Get ground truth keypoints
        target_keypoints = to_numpy(targets['gt_keypoints'])

        mpjpe = mpjpe_func(pred_keypoints, target_keypoints, reduce=True)
        return mpjpe


class SMPL_PAMPJPE:
    """
    Procrustes-Aligned Mean Per Joint Position Error for SMPL predictions.
    
    Converts SMPL parameters to keypoints, then evaluates after alignment.
    Expects predictions dict with 'pred_smpl' key from SMPLHead.predict().
    """
    
    def __init__(self, smpl_model_path='weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', affix=None):
        """
        Args:
            smpl_model_path: Path to SMPL model file (.pkl)
            affix: Optional suffix for metric name
        """
        self.smpl_model_path = smpl_model_path
        self.affix = affix
        self.name = f'smpl_pampjpe_{affix}' if affix is not None else 'smpl_pampjpe'
        self._smpl_model = None
    
    @property
    def smpl_model(self):
        """Lazy initialization of SMPL model."""
        if self._smpl_model is None:
            from models.smpl import SMPL
            self._smpl_model = SMPL(model_path=self.smpl_model_path)
            self._smpl_model = self._smpl_model.cuda()
        return self._smpl_model

    def __call__(self, preds, targets):
        # Extract SMPL parameters from prediction
        pred_smpl = preds.get('pred_smpl', None)
        if pred_smpl is None:
            raise ValueError("'pred_smpl' not found in predictions. Make sure you're using SMPLHead.")
        
        # Convert SMPL parameters to keypoints
        pred_keypoints = smpl_to_keypoints(pred_smpl, self.smpl_model, device='cuda')
        
        # Get ground truth keypoints
        target_keypoints = to_numpy(targets['gt_keypoints'])

        pampjpe = pampjpe_func(pred_keypoints, target_keypoints, reduce=True)
        return pampjpe


def _extract_smpl_global_orient(source, source_name):
    if source is None:
        raise ValueError(f"Missing {source_name}, cannot compute SMPL_PCMPJPE.")

    if isinstance(source, dict):
        if "global_orient" not in source:
            raise ValueError(f"{source_name} is missing required key 'global_orient'.")
        orient = source["global_orient"]
        return to_numpy(orient)

    if isinstance(source, list):
        if len(source) == 0:
            raise ValueError(f"{source_name} list is empty.")
        orient_list = []
        for idx, item in enumerate(source):
            if not isinstance(item, dict) or "global_orient" not in item:
                raise ValueError(
                    f"{source_name}[{idx}] must be a dict containing 'global_orient' to compute SMPL_PCMPJPE."
                )
            orient_list.append(np.asarray(to_numpy(item["global_orient"]), dtype=np.float32).reshape(3))
        return np.stack(orient_list, axis=0).astype(np.float32)

    arr = np.asarray(to_numpy(source), dtype=np.float32)
    if arr.shape[-1] < 3:
        raise ValueError(f"{source_name} must end with dim>=3 to parse root orientation, got {arr.shape}.")
    return arr[..., :3].astype(np.float32)


class SMPL_PCMPJPE:
    """
    Pelvis-centered MPJPE with pelvis orientation alignment for SMPL predictions.
    """

    def __init__(self, affix=None, pelvis_idx=0):
        self.affix = affix
        self.pelvis_idx = pelvis_idx
        self.name = f'smpl_pcmpjpe_{affix}' if affix is not None else 'smpl_pcmpjpe'

    def __call__(self, preds, targets):
        pred_keypoints = preds.get('pred_smpl_keypoints', None)
        if pred_keypoints is None:
            raise ValueError("'pred_smpl_keypoints' not found in predictions for SMPL_PCMPJPE.")
        pred_keypoints = to_numpy(pred_keypoints)
        target_keypoints = to_numpy(targets['gt_keypoints'])

        pred_smpl = preds.get('pred_smpl', None)
        if pred_smpl is not None:
            pred_root_rot = _extract_smpl_global_orient(pred_smpl, "pred_smpl")
        else:
            pred_smpl_params = preds.get('pred_smpl_params', None)
            pred_root_rot = _extract_smpl_global_orient(pred_smpl_params, "pred_smpl_params")

        gt_smpl = targets.get('gt_smpl', None)
        if gt_smpl is not None:
            gt_root_rot = _extract_smpl_global_orient(gt_smpl, "gt_smpl")
        else:
            gt_smpl_params = targets.get('gt_smpl_params', None)
            gt_root_rot = _extract_smpl_global_orient(gt_smpl_params, "gt_smpl_params")

        pcmpjpe = pcmpjpe_func(
            pred_keypoints,
            target_keypoints,
            pelvis_idx=self.pelvis_idx,
            pred_root_rot=pred_root_rot,
            gt_root_rot=gt_root_rot,
            reduce=True,
        )
        return pcmpjpe


class SMPL_VertexError:
    """
    Mean Per Vertex Position Error for SMPL mesh predictions.
    
    Converts SMPL parameters to vertices, then evaluates.
    Expects predictions dict with 'pred_smpl' key from SMPLHead.predict().
    """
    
    def __init__(self, smpl_model_path='weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', affix=None, scale_mm=1000.0):
        """
        Args:
            smpl_model_path: Path to SMPL model file (.pkl)
            affix: Optional suffix for metric name
            scale_mm: Multiplier to convert to millimeters (default: 1000.0)
        """
        self.smpl_model_path = smpl_model_path
        self.affix = affix
        self.scale_mm = scale_mm
        self.name = f'smpl_vertex_error_{affix}' if affix is not None else 'smpl_vertex_error'
        self._smpl_model = None
    
    @property
    def smpl_model(self):
        """Lazy initialization of SMPL model."""
        if self._smpl_model is None:
            from models.smpl import SMPL
            self._smpl_model = SMPL(model_path=self.smpl_model_path)
            self._smpl_model = self._smpl_model.cuda()
        return self._smpl_model

    def __call__(self, preds, targets):
        # Extract SMPL parameters from prediction
        pred_smpl = preds.get('pred_smpl', None)
        if pred_smpl is None:
            return float('nan')
        
        # Convert SMPL parameters to vertices
        pred_vertices = smpl_to_vertices(pred_smpl, self.smpl_model, device='cuda')
        
        # Get ground truth vertices (if available)
        target_vertices = targets.get('gt_vertices', None)
        if target_vertices is None:
            # Return NaN if ground truth vertices not provided
            return float('nan')
        
        target_vertices = to_numpy(target_vertices)
        
        # Compute L2 error per vertex
        vertex_error = np.sqrt(np.sum(np.square(pred_vertices - target_vertices), axis=-1))
        
        # Scale to mm and return mean
        mean_error = np.mean(vertex_error) * self.scale_mm
        return mean_error


class SMPL_PAVertexError:
    """
    Procrustes-Aligned Mean Per Vertex Position Error.
    
    Converts SMPL parameters to vertices, then evaluates after alignment.
    """
    
    def __init__(self, smpl_model_path='weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', affix=None, scale_mm=1000.0):
        """
        Args:
            smpl_model_path: Path to SMPL model file (.pkl)
            affix: Optional suffix for metric name
            scale_mm: Multiplier to convert to millimeters (default: 1000.0)
        """
        self.smpl_model_path = smpl_model_path
        self.affix = affix
        self.scale_mm = scale_mm
        self.name = f'smpl_pa_vertex_error_{affix}' if affix is not None else 'smpl_pa_vertex_error'
        self._smpl_model = None
    
    @property
    def smpl_model(self):
        """Lazy initialization of SMPL model."""
        if self._smpl_model is None:
            from models.smpl import SMPL
            self._smpl_model = SMPL(model_path=self.smpl_model_path)
            self._smpl_model = self._smpl_model.cuda()
        return self._smpl_model

    def __call__(self, preds, targets):
        from .mpjpe import compute_similarity_transform
        
        # Extract SMPL parameters from prediction
        pred_smpl = preds.get('pred_smpl', None)
        if pred_smpl is None:
            return float('nan')
        
        # Convert SMPL parameters to vertices
        pred_vertices = smpl_to_vertices(pred_smpl, self.smpl_model, device='cuda')
        
        # Get ground truth vertices
        target_vertices = targets.get('gt_vertices', None)
        if target_vertices is None:
            return float('nan')
        
        target_vertices = to_numpy(target_vertices)
        
        B = pred_vertices.shape[0]
        num_vertices = pred_vertices.shape[1]
        
        pa_vertex_error = np.zeros([B, num_vertices])
        
        for b in range(B):
            # Apply Procrustes alignment
            _, aligned_pred, _, _, _ = compute_similarity_transform(
                target_vertices[b], pred_vertices[b], compute_optimal_scale=True
            )
            # Compute error after alignment
            pa_vertex_error[b] = np.sqrt(np.sum(np.square(aligned_pred - target_vertices[b]), axis=1))
        
        # Scale to mm and return mean
        mean_error = np.mean(pa_vertex_error) * self.scale_mm
        return mean_error


class SMPL_ParamError:
    """
    Mean Absolute Error for SMPL parameters (pose, shape, translation).
    
    Directly evaluates predicted SMPL parameters against ground truth.
    Useful for debugging and understanding which parameters are difficult to predict.
    """
    
    def __init__(self, param_type='all', affix=None):
        """
        Args:
            param_type: Which parameters to evaluate - 'all', 'pose', 'shape', or 'transl'
            affix: Optional suffix for metric name
        """
        self.param_type = param_type
        self.affix = affix
        base_name = f'smpl_{param_type}_error'
        self.name = f'{base_name}_{affix}' if affix is not None else base_name

    def __call__(self, preds, targets):
        # Extract SMPL parameters from prediction dictionary
        pred_smpl = preds.get('pred_smpl', {})
        gt_smpl = targets.get('gt_smpl', {})
        
        if not pred_smpl or not gt_smpl:
            return float('nan')
        
        errors = []
        
        if self.param_type in ['all', 'pose']:
            # Pose includes global_orient + body_pose
            pred_global = to_numpy(pred_smpl['global_orient'])
            pred_body = to_numpy(pred_smpl['body_pose'])
            
            # Handle list of dicts or dict of tensors
            if isinstance(gt_smpl, list):
                gt_global = np.stack([
                    s['global_orient'] if isinstance(s['global_orient'], np.ndarray) 
                    else to_numpy(s['global_orient']) for s in gt_smpl
                ])
                gt_body = np.stack([
                    s['body_pose'] if isinstance(s['body_pose'], np.ndarray)
                    else to_numpy(s['body_pose']) for s in gt_smpl
                ])
            else:
                gt_global = to_numpy(gt_smpl['global_orient'])
                gt_body = to_numpy(gt_smpl['body_pose'])
            
            errors.append(np.abs(pred_global - gt_global))
            errors.append(np.abs(pred_body - gt_body))
        
        if self.param_type in ['all', 'shape']:
            pred_betas = to_numpy(pred_smpl['betas'])
            if isinstance(gt_smpl, list):
                gt_betas = np.stack([
                    s['betas'] if isinstance(s['betas'], np.ndarray)
                    else to_numpy(s['betas']) for s in gt_smpl
                ])
            else:
                gt_betas = to_numpy(gt_smpl['betas'])
            errors.append(np.abs(pred_betas - gt_betas))
        
        if self.param_type in ['all', 'transl']:
            pred_transl = to_numpy(pred_smpl['transl'])
            if isinstance(gt_smpl, list):
                gt_transl = np.stack([
                    s['transl'] if isinstance(s['transl'], np.ndarray)
                    else to_numpy(s['transl']) for s in gt_smpl
                ])
            else:
                gt_transl = to_numpy(gt_smpl['transl'])
            errors.append(np.abs(pred_transl - gt_transl))
        
        if not errors:
            return float('nan')
        
        # Concatenate all errors and compute mean
        all_errors = np.concatenate([e.flatten() for e in errors])
        return np.mean(all_errors)
