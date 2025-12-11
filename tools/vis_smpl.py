import cv2
from mmengine import load
import numpy as np
import torch
import os
import os.path as osp
import rerun as rr
import time

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models import SMPL

def load_pred_file(pred_file, pred_smpl_file):
    preds = load(pred_file)
    preds_smpl = load(pred_smpl_file)
    min_len = len(preds['sample_ids'])

    for key in preds_smpl:
        if (key not in preds or preds[key] is None) and preds_smpl[key] is not None:
            preds[key] = preds_smpl[key]
            min_len = min(min_len, len(preds[key]))
    for key in preds:
        if preds[key] is not None:
            preds[key] = preds[key][:min_len]

    sorted_indices = np.argsort(preds['sample_ids'])
    preds['sample_ids'] = [preds['sample_ids'][i] for i in sorted_indices]

    for key in ['pred_keypoints', 'pred_pose', 'pred_beta', 'pred_center',
        'gt_keypoints', 'gt_pose', 'gt_beta', 'gt_center']:
        if key in preds and preds[key] is not None:
            preds[key] = preds[key][sorted_indices]

    return preds

def get_verts(smpl_model: SMPL, pose, beta, center=None):
    smpl_model.eval()
    with torch.no_grad():
        verts, _ = smpl_model(torch.from_numpy(pose).float(), 
                              torch.from_numpy(beta).float(), 
                              torch.from_numpy(center).float() if center is not None else None)
    return verts.cpu().numpy()

def id_to_file_name(sample_id):
    E, S, A, idx = sample_id.split('_')
    idx = int(idx)
    file_name = f'{E}_{S}_{A}_frame{(idx+1):03d}'
    return file_name

def get_input_data(sample_id, dataset='mmfi_preproc', data_root='data/mmfi'):
    match dataset:
        case 'mmfi_preproc':
            fn = id_to_file_name(sample_id)
            rgb_path = osp.join(data_root, 'rgb', f'{fn}.jpg')
            depth_path = osp.join(data_root, 'depth', f'{fn}.jpg')
            lidar_path = osp.join(data_root, 'lidar', f'{fn}.npy')
            mmwave_path = osp.join(data_root, 'mmwave', f'{fn}.npy')
            
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            lidar = np.load(lidar_path)
            mmwave = np.load(mmwave_path)
            return rgb, depth, lidar, mmwave
        case _:
            raise NotImplementedError(f'Unknown dataset: {dataset}')

def process_batch(smpl_model: SMPL, batch, meta, dataset='mmfi', data_root='data/mmfi'):
    batch_ids = batch['sample_ids']
    batch_gt_pose = batch['gt_pose']
    batch_gt_beta = batch['gt_beta']
    batch_gt_kps = batch['gt_keypoints']
    batch_pred_pose = batch['pred_pose']
    batch_pred_beta = batch['pred_beta']
    batch_pred_kps = batch['pred_keypoints']

    batch_gt_verts = get_verts(smpl_model, batch_gt_pose, batch_gt_beta)
    batch_gt_verts /= meta['gt_rot'][np.newaxis, np.newaxis, :]
    batch_gt_verts += batch['gt_center'][:, np.newaxis, :]
    batch_pred_verts = get_verts(smpl_model, batch_pred_pose, batch_pred_beta)
    batch_pred_verts /= meta['pred_rot'][np.newaxis, np.newaxis, :]
    batch_pred_verts += batch['pred_center'][:, np.newaxis, :]

    input_data = {'rgb': [], 'depth': [], 'lidar': [], 'mmwave': []}
    for id in batch_ids:
        rgb, depth, lidar, mmwave = get_input_data(id, dataset, data_root)
        input_data['rgb'].append(rgb)
        input_data['depth'].append(depth)
        input_data['lidar'].append(lidar)
        input_data['mmwave'].append(mmwave)
    
    output_data = {
        'gt_verts': batch_gt_verts,
        'gt_kps': batch_gt_kps,
        'pred_verts': batch_pred_verts,
        'pred_kps': batch_pred_kps,
    }

    return input_data, output_data

def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals for a mesh."""
    # Initialize normals array
    normals = np.zeros_like(vertices)
    
    # Get triangle vertices
    tris = vertices[faces]
    
    # Compute face normals
    v0 = tris[:, 1] - tris[:, 0]
    v1 = tris[:, 2] - tris[:, 0]
    face_normals = np.cross(v0, v1)
    
    # Accumulate face normals to vertices
    for i in range(3):
        normals[faces[:, i]] += face_normals
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-10)
    
    return normals

def vis_batch_in_rerun(input_data, output_data, faces, port=8097):
    rr.init("mmhpe_visualization")
    server_uri = rr.serve_grpc(grpc_port=port+1)
    rr.serve_web_viewer(web_port=port, open_browser=False, connect_to=server_uri)

    rr.log("gt_smpl_world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    rr.log("pred_smpl_world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    rr.log("lidar_world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    rr.log("mmwave_world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

    # Set a nice skin-tone color
    num_verts = output_data['gt_verts'][0].shape[0]
    vertex_colors = np.ones((num_verts, 4), dtype=np.uint8)
    vertex_colors[:, :3] = [220, 180, 150]  # RGB skin tone
    vertex_colors[:, 3] = 255  # Alpha

    for frame_idx in range(len(output_data['gt_verts'])):
        rr.set_time(timeline='time', sequence=frame_idx)

        # Compute normals for this frame
        gt_normals = compute_vertex_normals(output_data['gt_verts'][frame_idx], faces)
        pred_normals = compute_vertex_normals(output_data['pred_verts'][frame_idx], faces)
        
        # Log mesh with normals
        rr.log("gt_smpl_world/smpl_mesh", rr.Mesh3D(
            vertex_positions=output_data['gt_verts'][frame_idx],
            triangle_indices=faces,
            vertex_normals=gt_normals,
            vertex_colors=vertex_colors,
            albedo_factor=[0.8, 0.8, 0.8, 1.0],
        ))
        rr.log("gt_smpl_world/gt_keypoints", rr.Points3D(output_data['gt_kps'][frame_idx][:, :3]))

        rr.log("pred_smpl_world/smpl_mesh", rr.Mesh3D(
            vertex_positions=output_data['pred_verts'][frame_idx],
            triangle_indices=faces,
            vertex_normals=pred_normals,
            vertex_colors=vertex_colors,
            albedo_factor=[0.8, 0.8, 0.8, 1.0],
        ))
        rr.log("pred_smpl_world/pred_keypoints", rr.Points3D(output_data['pred_kps'][frame_idx][:, :3]))
        
        # Log images
        rr.log("rgb", rr.Image(input_data['rgb'][frame_idx]))
        rr.log("depth", rr.Image(input_data['depth'][frame_idx]))
        
        # Log point clouds
        rr.log("lidar_world/lidar", rr.Points3D(input_data['lidar'][frame_idx]))
        rr.log("mmwave_world/mmwave", rr.Points3D(input_data['mmwave'][frame_idx][:, :3]))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down visualization server.")

if __name__ == '__main__':
    pred_file = '/home/zpengac/mmhpe/MmMvHPE/logs/dev/20251210_001128/TestModel_1208_test_predictions.pkl'
    pred_smpl_file = '/home/zpengac/mmhpe/MmMvHPE/logs/dev/20251210_001128/TestModel_1208_test_predictions_smpl2.pkl'
    data_root = '/home/zpengac/mmhpe/MmMvHPE/data/mmfi'
    dataset = 'mmfi_preproc'

    preds = load_pred_file(pred_file, pred_smpl_file)

    smpl_model = SMPL(model_path='/home/zpengac/mmhpe/MmMvHPE/weights/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')
    faces = smpl_model.th_faces.numpy()

    batch_size = 32
    batch = {}
    meta = {}
    for key in preds:
        match key:
            case 'gt_rot' | 'pred_rot':
                meta[key] = preds[key]
            case _ if preds[key] is not None:
                batch[key] = preds[key][0:batch_size]

    input_data, output_data = process_batch(
        smpl_model,
        batch,
        meta,
        dataset=dataset,
        data_root=data_root,
    )

    vis_batch_in_rerun(input_data, output_data, faces, port=8097)