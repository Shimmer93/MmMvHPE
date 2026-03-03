from __future__ import annotations

import numpy as np
import torch

from misc.skeleton import COCOSkeleton, H36MSkeleton, SMPLSkeleton


def get_skeleton_class(skeleton_format: str):
    skeleton_map = {
        "smpl": SMPLSkeleton,
        "h36m": H36MSkeleton,
        "coco": COCOSkeleton,
    }
    return skeleton_map.get(skeleton_format, SMPLSkeleton)


def load_smpl_model(smpl_model_path: str, device: str = "cuda"):
    from models.smpl import SMPL

    smpl_model = SMPL(model_path=smpl_model_path)
    smpl_model = smpl_model.to(device)
    smpl_model.eval()
    return smpl_model


def smpl_params_to_mesh(smpl_model, smpl_params, device: str = "cuda"):
    if isinstance(smpl_params["global_orient"], np.ndarray):
        global_orient = torch.from_numpy(smpl_params["global_orient"]).float().to(device)
        body_pose = torch.from_numpy(smpl_params["body_pose"]).float().to(device)
        betas = torch.from_numpy(smpl_params["betas"]).float().to(device)
        transl = torch.from_numpy(smpl_params["transl"]).float().to(device)
    else:
        global_orient = smpl_params["global_orient"].to(device)
        body_pose = smpl_params["body_pose"].to(device)
        betas = smpl_params["betas"].to(device)
        transl = smpl_params["transl"].to(device)

    if global_orient.dim() == 1:
        global_orient = global_orient.unsqueeze(0)
        body_pose = body_pose.unsqueeze(0)
        betas = betas.unsqueeze(0)
        transl = transl.unsqueeze(0)

    batch_size = global_orient.shape[0]
    pose = torch.cat([global_orient, body_pose], dim=1)
    expected_betas = smpl_model.th_betas.shape[1]
    if betas.shape[1] < expected_betas:
        betas_padded = torch.zeros(batch_size, expected_betas, device=device, dtype=betas.dtype)
        betas_padded[:, : betas.shape[1]] = betas
        betas = betas_padded

    with torch.no_grad():
        vertices, joints = smpl_model(pose, betas, transl)

    return vertices.cpu().numpy(), joints.cpu().numpy(), smpl_model.th_faces.cpu().numpy()


def canonicalize_smpl_params(smpl_params: dict) -> dict:
    canonical = {}
    for key, val in smpl_params.items():
        if key in ("global_orient", "transl"):
            canonical[key] = torch.zeros_like(val) if isinstance(val, torch.Tensor) else np.zeros_like(val)
        else:
            canonical[key] = val
    return canonical


def split_smpl_params(smpl_params_vector):
    params = smpl_params_vector if isinstance(smpl_params_vector, torch.Tensor) else np.asarray(smpl_params_vector)
    if params.shape[-1] < 82:
        raise ValueError(f"Expected SMPL params with 82 dims, got {params.shape[-1]}.")
    pose = params[..., :72]
    betas = params[..., 72:82]
    if isinstance(params, torch.Tensor):
        zeros = torch.zeros((*params.shape[:-1], 3), device=params.device, dtype=params.dtype)
    else:
        zeros = np.zeros((*params.shape[:-1], 3), dtype=params.dtype)
    return {
        "global_orient": pose[..., :3],
        "body_pose": pose[..., 3:72],
        "betas": betas,
        "transl": zeros,
    }

