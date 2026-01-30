import cv2
import numpy as np
import torch
import os
import os.path as osp
import json
import re
import rerun as rr
import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
try:
    from segment_anything import sam_model_registry, SamPredictor
    from ultralytics import YOLO
    # SAM_AVAILABLE = True
except ImportError:
    print("Please install segment_anything and ultralytics packages to use the masking functionality.")

SAM_AVAILABLE = False

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models.smpl import SMPL
from misc.skeleton import H36MSkeleton, SMPLSkeleton
from misc.utils import load

def load_pred_file(pred_file, pred_smpl_file):
    preds = load(pred_file)
    preds_smpl = load(pred_smpl_file)
    min_len = len(preds['sample_ids'])

    for key in preds_smpl:
        if (key not in preds or preds[key] is None) and preds_smpl[key] is not None:
            preds[key] = preds_smpl[key]
    #         min_len = min(min_len, len(preds[key]))
    # for key in preds:
    #     if preds[key] is not None:
    #         preds[key] = preds[key][:min_len]

    # sorted_indices = np.argsort(preds['sample_ids'])
    # preds['sample_ids'] = [preds['sample_ids'][i] for i in sorted_indices]

    # for key in ['pred_keypoints', 'pred_pose', 'pred_beta', 'pred_center',
    #     'gt_keypoints', 'gt_pose', 'gt_beta', 'gt_center']:
    #     if key in preds and preds[key] is not None:
    #         preds[key] = preds[key][sorted_indices]

    return preds

def load_pred_file_new(pred_file):
    preds = load(pred_file)
    preds['pred_keypoints'] = preds['pred_smpl_keypoints']
    smpl_params = preds['pred_smpl_params']
    preds['pred_pose'] = smpl_params[:, 3:72]
    preds['pred_global_orient'] = np.zeros((smpl_params.shape[0], 3))
    preds['pred_beta'] = smpl_params[:, 72:82]
    preds['pred_translation'] = np.zeros((smpl_params.shape[0], 3))
    preds['gt_pose'] = preds['gt_smpl_params'][:, 3:72]
    preds['gt_global_orient'] = np.zeros((smpl_params.shape[0], 3))
    preds['gt_beta'] = preds['gt_smpl_params'][:, 72:82]
    preds['gt_translation'] = np.zeros((smpl_params.shape[0], 3))
    return preds


def get_verts(smpl_model, pose, global_orient, beta, translation, center=None):
    smpl_model.eval()
    with torch.no_grad():
        # pose is body pose (batch, 69) in axis-angle
        pose_tensor = torch.from_numpy(pose).float()
        global_orient_tensor = torch.from_numpy(global_orient).float()
        full_pose = torch.cat([global_orient_tensor, pose_tensor], dim=1)
        beta_tensor = torch.from_numpy(beta).float()
        translation_tensor = torch.from_numpy(translation).float()

        verts, _ = smpl_model(
            th_pose_axisang=full_pose,
            th_betas=beta_tensor,
            th_trans=translation_tensor,
        )
    return verts.cpu().numpy()

def id_to_file_name(sample_id):
    E, S, A, idx = sample_id.split('_')
    idx = int(idx)
    file_name = f'{E}_{S}_{A}_frame{(idx+1):03d}'
    return file_name

_HUMMAN_ID_RE = re.compile(
    r"^(?P<seq>p\d+_a\d+)_rgb_(?P<rgb>kinect_\d{3}|iphone)_depth_"
    r"(?P<depth>kinect_\d{3}|iphone)_(?P<frame>\d+)$"
)

def id_to_file_name_humman(sample_id, modality):
    match = _HUMMAN_ID_RE.match(sample_id)
    if match is None:
        raise ValueError(f"Unexpected HUMMAN sample_id format: {sample_id}")
    frame_idx = int(match.group("frame"))
    frame_token = f"{frame_idx+1:06d}"
    if modality == "rgb":
        camera = match.group("rgb")
    elif modality == "depth":
        camera = match.group("depth")
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return f"{match.group('seq')}_{camera}_{frame_token}"

def depth_to_lidar_frame(depth, K, R, T, min_depth=1e-6):
    K_inv = np.linalg.inv(K)
    H, W = depth.shape
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.reshape(-1)
    valid = z > min_depth
    pixels = np.stack([xmap.reshape(-1), ymap.reshape(-1), np.ones(H * W)], axis=0)
    rays = K_inv @ pixels
    cam_points = rays * z
    cam_points = cam_points[:, valid]
    world_points = (R.T @ (cam_points - T)).T
    return world_points.astype(np.float32)

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
        case 'humman_preproc':
            rgb_fn = id_to_file_name_humman(sample_id, modality="rgb")
            depth_fn = id_to_file_name_humman(sample_id, modality="depth")
            rgb_path = osp.join(data_root, 'rgb', f'{rgb_fn}.jpg')
            depth_path = osp.join(data_root, 'depth', f'{depth_fn}.png')
            print(sample_id, depth_path)
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0  # convert mm to meters
            lidar = None
            match = _HUMMAN_ID_RE.match(sample_id)
            if match is not None:
                seq_name = match.group("seq")
                depth_cam = match.group("depth")
                camera_file = osp.join(data_root, "cameras", f"{seq_name}_cameras.json")
                with open(camera_file, "r") as f:
                    cameras = json.load(f)
                if depth_cam.startswith("kinect"):
                    cam_key = f"kinect_depth_{depth_cam.split('_')[1]}"
                else:
                    cam_key = "iphone"
                if cam_key in cameras:
                    cam_params = cameras[cam_key]
                    K = np.array(cam_params["K"], dtype=np.float32)
                    R = np.array(cam_params["R"], dtype=np.float32)
                    T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
                    lidar = depth_to_lidar_frame(depth, K, R, T)

            # repeat depth to make it 3-channel
            depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
            return rgb, depth, lidar, None
        case _:
            raise NotImplementedError(f'Unknown dataset: {dataset}')

def mask_image(batch_imgs, model_type='vit_b', model_path='weights/sam_vit_b_01ec64.pth'):
    # batch_imgs: B x H x W x 3 (RGB) numpy array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Models
    yolo_model = YOLO('yolov8n.pt')  # Pretrained YOLOv8n model

    sam_checkpoint = model_path
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    batch_masks = []
    for img in tqdm(batch_imgs):
        # 2. Run YOLO Detection
        results = yolo_model(img, classes=[0], max_det=1, verbose=False)  # Person class only

        if len(results[0].boxes) == 0:
            # No person detected, return empty box
            box = np.array([0, 0, img.shape[1], img.shape[0]])
        else:
            box = results[0].boxes.xyxy.cpu().numpy()[0]  # Get the best box

        # 3. Generate Mask with SAM
        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            box=box,
            multimask_output=False
        )
        single_mask = masks[0]  # Shape: [H, W]
        batch_masks.append(single_mask)

    batch_masks = np.stack(batch_masks, axis=0)  # B x H x W
    
    # keep only the person region
    new_imgs = batch_imgs * batch_masks[:, :, :, np.newaxis]

    return new_imgs

def process_batch(smpl_model, batch, meta, dataset='mmfi', data_root='data/mmfi'):
    batch_ids = batch['sample_ids']
    batch_gt_pose = batch['gt_pose']
    batch_gt_global_orient = batch['pred_global_orient']
    batch_gt_beta = batch['gt_beta']
    batch_gt_translation = batch['gt_translation']
    batch_gt_kps = batch['gt_keypoints']
    batch_pred_pose = batch['pred_pose']
    batch_pred_global_orient = batch['pred_global_orient']
    batch_pred_beta = batch['pred_beta']
    batch_pred_translation = batch['pred_translation']
    batch_pred_kps = batch['pred_keypoints']

    batch_gt_translation += batch_gt_kps[:, 0, :] + np.array([0., 0.2, 0.])  # add pelvis position back
    # batch_pred_translation += batch_pred_kps[:, 0, :] + np.array([0., 0.1, 0.])  # add pelvis position back
    batch_gt_verts = get_verts(smpl_model, batch_gt_pose, batch_gt_global_orient, batch_gt_beta, batch_gt_translation)
    # batch_gt_verts /= meta['gt_rot'][np.newaxis, np.newaxis, :]
    # batch_gt_verts += batch['gt_center'][:, np.newaxis, :]
    batch_pred_verts = get_verts(smpl_model, batch_pred_pose, batch_pred_global_orient, batch_pred_beta, batch_pred_translation)
    # batch_pred_verts /= meta['pred_rot'][np.newaxis, np.newaxis, :]
    # batch_pred_verts += batch['pred_center'][:, np.newaxis, :]
    # batch_pred_verts[..., 1] -= 0.2  # lift up a bit for better visualization
    # batch_gt_verts *= np.array([1., -1, -1.])
    # batch_pred_verts *= np.array([1., -1, -1.])
    # batch_gt_kps *= np.array([1., -1, -1.])
    # batch_pred_kps *= np.array([1., -1, -1.])


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

def compute_edge_segments(kps, edges):
    # kps: B x N x 3
    # edges: M x 2
    edges = np.array(edges)
    batch_size = kps.shape[0]
    edge_segments = np.zeros((batch_size, edges.shape[0], 2, 3))
    # for i in range(batch_size):
    edge_segments[:, :, 0, :] = kps[:, edges[:, 0], :]
    edge_segments[:, :, 1, :] = kps[:, edges[:, 1], :]
    return edge_segments

def vis_batch_in_rerun(input_data, output_data, faces, edges, port=8097):
    rr.init("mmhpe_visualization")
    server_uri = rr.serve_grpc(grpc_port=port+1)
    rr.serve_web_viewer(web_port=port, open_browser=False, connect_to=server_uri)

    rr.log("gt_smpl_world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    rr.log("pred_smpl_world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    rr.log("lidar_world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    # rr.log("mmwave_world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    if SAM_AVAILABLE:
        segs = mask_image(np.array(input_data['rgb']))

    gt_bones = compute_edge_segments(output_data['gt_kps'], edges)
    pred_bones = compute_edge_segments(output_data['pred_kps'], edges)

    for frame_idx in range(len(output_data['gt_verts'])):
        rr.set_time(timeline='time', sequence=frame_idx)

        # Log images
        rr.log("rgb", rr.Image(input_data['rgb'][frame_idx]))
        rr.log("depth", rr.Image(input_data['depth'][frame_idx]))
        if SAM_AVAILABLE:
            rr.log("segmented_rgb", rr.Image(segs[frame_idx]))
        
        # Log point clouds
        rr.log("lidar_world/lidar", rr.Points3D(input_data['lidar'][frame_idx]))
        # rr.log("mmwave_world/mmwave", rr.Points3D(input_data['mmwave'][frame_idx][:, :3], radii=0.03, colors=[225,184,230]))

        # Compute normals for this frame
        gt_normals = compute_vertex_normals(output_data['gt_verts'][frame_idx], faces)
        pred_normals = compute_vertex_normals(output_data['pred_verts'][frame_idx], faces)
        
        # Log mesh with normals
        rr.log("gt_smpl_world/smpl_mesh", rr.Mesh3D(
            vertex_positions=output_data['gt_verts'][frame_idx],
            triangle_indices=faces,
            vertex_normals=gt_normals,
            albedo_factor=[0.86, 0.7, 0.59, 0.3],
        ))
        rr.log("gt_smpl_world/gt_keypoints", rr.Points3D(
            output_data['gt_kps'][frame_idx][:, :3],
            colors=[0, 180, 180],
            radii=0.02,
        ))

        rr.log("gt_smpl_world/gt_skeleton", rr.LineStrips3D(
            strips=gt_bones[frame_idx],
            colors=[0, 180, 180],
        ))

        rr.log("pred_smpl_world/smpl_mesh", rr.Mesh3D(
            vertex_positions=output_data['pred_verts'][frame_idx],
            triangle_indices=faces,
            vertex_normals=pred_normals,
            albedo_factor=[0.86, 0.7, 0.59, 0.3],
        ))
        rr.log("pred_smpl_world/pred_keypoints", rr.Points3D(
            output_data['pred_kps'][frame_idx][:, :3],
            colors=[0, 180, 180],
            radii=0.02,
        ))
        rr.log("pred_smpl_world/pred_skeleton", rr.LineStrips3D(
            strips=pred_bones[frame_idx],
            colors=[0, 180, 180],
        ))
        

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down visualization server.")

if __name__ == '__main__':
    pred_file = '/home/zpengac/mmhpe/MmMvHPE/logs/dev_humman_smpl/humman_smpl_token_fixed_mv_wcam5/HummanVIBEToken_test_predictions.pkl'
    # pred_smpl_file = '/home/zpengac/mmhpe/MmMvHPE/logs/dev/20251210_001128/TestModel_1208_test_predictions_smpl.pkl'
    data_root = '/opt/data/humman_cropped'
    dataset = 'humman_preproc'

    preds = load_pred_file_new(pred_file)
    # preds = load_pred_file(pred_file, pred_smpl_file)

    smpl_model = SMPL(model_path='/home/zpengac/mmhpe/MmMvHPE/weights/smpl/SMPL_NEUTRAL.pkl')
    # smpl_model = smpl_model = smplx.create('/home/zpengac/mmhpe/MmMvHPE/weights', 
    #                                         model_type='smpl', 
    #                                         gender='neutral', 
    #                                         use_face_contour=False,
    #                                         num_betas=10)
    faces = smpl_model.th_faces.cpu().numpy()
    print(f'Loaded SMPL model with {faces.shape[0]} faces.')
    # print(f"SMPL model loaded from {model_path}")
    edges = SMPLSkeleton.bones

    batch_size = 50
    i_batch = 0 # 0 90 160
    batch = {}
    meta = {}
    for key in preds:
        match key:
            case 'gt_rot' | 'pred_rot':
                meta[key] = preds[key]
            case _ if preds[key] is not None:
                batch[key] = preds[key][i_batch * batch_size : (i_batch + 1) * batch_size]

    input_data, output_data = process_batch(
        smpl_model,
        batch,
        meta,
        dataset=dataset,
        data_root=data_root,
    )

    vis_batch_in_rerun(input_data, output_data, faces, edges, port=8097)
