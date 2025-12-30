import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .skeleton import JOINT_COLOR_MAP, H36MSkeleton, COCOSkeleton, SMPLSkeleton

# SMPL model color constants
SMPL_MESH_COLOR = [0.65, 0.75, 0.85]  # Light blue for mesh
SMPL_MESH_EDGE_COLOR = [0.3, 0.4, 0.5]  # Darker blue for edges

def project_3d_to_2d(points_3d, intrinsic_matrix):
    """Project 3D points in camera coordinates to 2D image coordinates.
    
    Args:
        points_3d (np.ndarray): 3D points in camera coordinates, shape (N, 3)
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix, shape (3, 3)
    
    Returns:
        np.ndarray: 2D points in image coordinates, shape (N, 2)
    """
    # Homogeneous coordinates
    points_2d_homo = intrinsic_matrix @ points_3d.T  # (3, N)
    
    # Normalize by Z coordinate
    points_2d = points_2d_homo[:2, :] / points_2d_homo[2:3, :]  # (2, N)
    
    return points_2d.T  # (N, 2)

def denormalize(img_array, mean, std):
    """Denormalize an image array.

    Args:
        img_array (np.ndarray): Normalized image array of shape (H, W, C).
        mean (list or np.ndarray): Mean used for normalization.
        std (list or np.ndarray): Standard deviation used for normalization.

    Returns:
        np.ndarray: Denormalized image array.
    """
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    denorm_img = img_array * std + mean
    denorm_img = np.clip(denorm_img, 0, 255)
    return denorm_img.astype(np.uint8)

def reverse_affine_transform(points, affine_matrix):
    """
    Reverse the affine transformation applied to the points.

    Args:
        points (np.ndarray): Points of shape (N, 3) or (N, 2).
        affine_matrix (np.ndarray): Affine transformation matrix of shape (4, 4) or (3, 3).

    Returns:
        np.ndarray: Transformed points of the same shape as input.
    """
    num_dims = points.shape[1]
    if num_dims == 3:
        assert affine_matrix.shape == (4, 4), "Affine matrix must be of shape (4, 4) for 3D points."
        inv_affine = np.linalg.inv(affine_matrix)
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points_homogeneous = points_homogeneous @ inv_affine.T
        transformed_points = transformed_points_homogeneous[:, :3]
    elif num_dims == 2:
        assert affine_matrix.shape == (3, 3), "Affine matrix must be of shape (3, 3) for 2D points."
        inv_affine = np.linalg.inv(affine_matrix)
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points_homogeneous = points_homogeneous @ inv_affine.T
        transformed_points = transformed_points_homogeneous[:, :2]
    else:
        raise ValueError("Points must be either 2D or 3D.")
    return transformed_points

def get_bounds(points):
    all_points = points[..., :3].reshape(-1, 3)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]

def set_3d_ax_limits(ax, bounds, padding=0.1):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    ranges = [max_x - min_x, max_y - min_y, max_z - min_z]
    ax.set_box_aspect(ranges)
    ax.set_xlim(min_x - padding * ranges[0], max_x + padding * ranges[0])
    ax.set_ylim(min_y - padding * ranges[1], max_y + padding * ranges[1])
    ax.set_zlim(min_z - padding * ranges[2], max_z + padding * ranges[2])

def set_2d_ax_limits(ax, bounds, dims=[0, 1], padding=0.1):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    bounds_arr = np.array([min_x, max_x, min_y, max_y, min_z, max_z])
    ax.set_aspect('equal')
    
    min_vals = bounds_arr[np.array(dims) * 2]
    max_vals = bounds_arr[np.array(dims) * 2 + 1]
    ranges = max_vals - min_vals
    
    ax.set_xlim(min_vals[0] - padding * ranges[0], max_vals[0] + padding * ranges[0])
    ax.set_ylim(min_vals[1] - padding * ranges[1], max_vals[1] + padding * ranges[1])

def plot_2d_skeleton(ax, keypoints, dims=[0, 1], edges=None, color_map=JOINT_COLOR_MAP, linewidth=2, s=20):
    if edges is not None:
        for i, j in edges:
            ax.plot([keypoints[i, dims[0]], keypoints[j, dims[0]]],
                    [keypoints[i, dims[1]], keypoints[j, dims[1]]], color='0.5', linewidth=linewidth)
    
    for i, (x, y) in enumerate(keypoints[:, dims]):
        ax.scatter(x, y, color=color_map[i], marker='o', s=s)

def plot_3d_skeleton(ax, keypoints, edges=None, color_map=JOINT_COLOR_MAP, linewidth=2, s=20):
    if edges is not None:
        for i, j in edges:
            ax.plot([keypoints[i, 0], keypoints[j, 0]],
                    [keypoints[i, 1], keypoints[j, 1]],
                    [keypoints[i, 2], keypoints[j, 2]], color='0.5', linewidth=linewidth)
    
    for i, (x, y, z) in enumerate(keypoints):
        ax.scatter(x, y, z, color=color_map[i], marker='o', s=s)

def plot_2d_skeleton_on_image(ax, image, keypoints_2d, edges=None, color_map=JOINT_COLOR_MAP, linewidth=2, s=50):
    """Plot 2D skeleton overlay on an image.
    
    Args:
        ax: matplotlib axis
        image: RGB image array (H, W, 3), or None if image already shown on axis
        keypoints_2d: 2D keypoint coordinates (N, 2) in image space
        edges: list of (i, j) tuples representing skeleton bones
        color_map: color map for joints
        linewidth: line width for bones
        s: marker size for joints
    """
    if image is not None:
        ax.imshow(image)
    
    # Plot bones
    if edges is not None:
        for i, j in edges:
            ax.plot([keypoints_2d[i, 0], keypoints_2d[j, 0]],
                    [keypoints_2d[i, 1], keypoints_2d[j, 1]], 
                    color='lime', linewidth=linewidth, alpha=0.8)
    
    # Plot joints
    for i, (x, y) in enumerate(keypoints_2d):
        ax.scatter(x, y, color=color_map[i], marker='o', s=s, 
                  edgecolors='white', linewidths=1.5, zorder=10)
    
    ax.axis('off')


def get_smpl_joints_from_params(smpl_model, global_orient, body_pose, betas, transl, device='cuda'):
    """Get 3D joint positions from SMPL parameters."""
    single_sample = global_orient.ndim == 1
    if single_sample:
        global_orient = global_orient[np.newaxis, :]
        body_pose = body_pose[np.newaxis, :]
        betas = betas[np.newaxis, :]
        transl = transl[np.newaxis, :]
    
    N = global_orient.shape[0]
    pose = np.concatenate([global_orient, body_pose], axis=1)
    
    expected_betas = smpl_model.th_betas.shape[1]
    if betas.shape[1] < expected_betas:
        betas_padded = np.zeros((N, expected_betas), dtype=betas.dtype)
        betas_padded[:, :betas.shape[1]] = betas
        betas = betas_padded
    
    th_pose = torch.from_numpy(pose).float().to(device)
    th_betas = torch.from_numpy(betas).float().to(device)
    th_trans = torch.from_numpy(transl).float().to(device)
    
    smpl_model = smpl_model.to(device)
    smpl_model.eval()
    with torch.no_grad():
        verts, joints = smpl_model(th_pose, th_betas, th_trans)
    
    joints = joints.cpu().numpy()
    if single_sample:
        joints = joints[0]
    
    return joints


def get_smpl_mesh_from_params(smpl_model, global_orient, body_pose, betas, transl, device='cuda'):
    """Get mesh vertices and faces from SMPL parameters."""
    single_sample = global_orient.ndim == 1
    if single_sample:
        global_orient = global_orient[np.newaxis, :]
        body_pose = body_pose[np.newaxis, :]
        betas = betas[np.newaxis, :]
        transl = transl[np.newaxis, :]
    
    N = global_orient.shape[0]
    pose = np.concatenate([global_orient, body_pose], axis=1)
    
    expected_betas = smpl_model.th_betas.shape[1]
    if betas.shape[1] < expected_betas:
        betas_padded = np.zeros((N, expected_betas), dtype=betas.dtype)
        betas_padded[:, :betas.shape[1]] = betas
        betas = betas_padded
    
    th_pose = torch.from_numpy(pose).float().to(device)
    th_betas = torch.from_numpy(betas).float().to(device)
    th_trans = torch.from_numpy(transl).float().to(device)
    
    smpl_model = smpl_model.to(device)
    smpl_model.eval()
    with torch.no_grad():
        verts, joints = smpl_model(th_pose, th_betas, th_trans)
    
    faces = smpl_model.th_faces.cpu().numpy()
    vertices = verts.cpu().numpy()
    joints = joints.cpu().numpy()
    
    if single_sample:
        vertices = vertices[0]
        joints = joints[0]
    
    return vertices, faces, joints


def plot_smpl_mesh_3d(ax, vertices, faces, color=SMPL_MESH_COLOR, alpha=0.5, 
                      edge_color=None, linewidth=0.1):
    """Plot SMPL mesh in 3D."""
    mesh_faces = vertices[faces]
    mesh = Poly3DCollection(mesh_faces, alpha=alpha)
    mesh.set_facecolor(color)
    if edge_color is not None:
        mesh.set_edgecolor(edge_color)
        mesh.set_linewidth(linewidth)
    else:
        mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)
    return mesh


def plot_smpl_mesh_2d(ax, vertices, faces, intrinsic_matrix, color=SMPL_MESH_COLOR, 
                      alpha=0.3, edge_color=SMPL_MESH_EDGE_COLOR, linewidth=0.2):
    """Plot projected SMPL mesh in 2D."""
    vertices_2d = project_3d_to_2d(vertices, intrinsic_matrix)
    depths = vertices[:, 2]
    face_depths = depths[faces].mean(axis=1)
    sorted_indices = np.argsort(-face_depths)
    triangles = vertices_2d[faces[sorted_indices]]
    poly_collection = PolyCollection(
        triangles, 
        facecolors=[(*color, alpha)] * len(triangles),
        edgecolors=edge_color,
        linewidths=linewidth
    )
    ax.add_collection(poly_collection)
    return poly_collection


def plot_smpl_mesh_on_image(ax, image, vertices, faces, intrinsic_matrix,
                            color=SMPL_MESH_COLOR, alpha=0.4, 
                            edge_color=None, linewidth=0.1):
    """Plot projected SMPL mesh overlaid on an image."""
    ax.imshow(image)
    plot_smpl_mesh_2d(ax, vertices, faces, intrinsic_matrix, 
                      color=color, alpha=alpha, edge_color=edge_color, linewidth=linewidth)
    ax.axis('off')


def plot_2d_point_cloud(ax, points, dims=[0, 1], color='k', s=1):
    points_ = points[..., :3].reshape(-1, 3)
    ax.scatter(points_[:, dims[0]], points_[:, dims[1]], c=color, s=s)

def plot_3d_point_cloud(ax, points, color='k', s=1):
    points_ = points[..., :3].reshape(-1, 3)
    ax.scatter(points_[:, 0], points_[:, 1], points_[:, 2], c=color, s=s)

def visualize_multimodal_sample(batch, pred_dict, skl_format=None, denorm_params=None, smpl_model=None, device='cuda'):
    # Plot the following subplots:
    # Row 1: RGB, Depth, LiDAR, mmWave (if not available, leave blank)
    # Row 2: GT 3D skeleton, GT XY, GT XZ, GT YZ
    # Row 3: Pred 3D skeleton, Pred XY, Pred XZ, Pred YZ
    # Row 4: RGB+GT overlay, Depth+GT overlay, RGB+Pred overlay, Depth+Pred overlay

    bones = None
    if skl_format == 'h36m':
        bones = H36MSkeleton.bones
    elif skl_format == 'coco':
        bones = COCOSkeleton.bones
    elif skl_format == 'smpl':
        bones = SMPLSkeleton.bones

    fig = plt.figure(figsize=(16, 16))
    axes = []
    is_3d = [
        False, False, True, True,
        True, False, False, False,
        True, False, False, False,
        False, False, False, False
    ]

    for i in range(16):
        ax = fig.add_subplot(4, 4, i+1, projection='3d' if is_3d[i] else None)
        axes.append(ax)

    sample_idx = 0  # visualize the first sample in the batch
    plot_idx = 0
    
    # Store RGB and depth images for later overlay
    rgb_image = None
    depth_image = None

    # RGB image
    if 'input_rgb' in batch and batch['input_rgb'] is not None:
        rgb_image = batch['input_rgb'][sample_idx][-1].permute(1, 2, 0).cpu().numpy()
        if denorm_params is not None:
            rgb_image = denormalize(rgb_image, denorm_params['rgb_mean'], denorm_params['rgb_std'])
        axes[plot_idx].imshow(rgb_image)
        axes[plot_idx].set_title('RGB Image')
        axes[plot_idx].axis('off')
    plot_idx += 1
    # Depth image
    if 'input_depth' in batch and batch['input_depth'] is not None:
        depth_image = batch['input_depth'][sample_idx][-1].permute(1, 2, 0).cpu().numpy()
        if denorm_params is not None:
            depth_image = denormalize(depth_image, denorm_params['depth_mean'], denorm_params['depth_std'])
        
        # Handle both single-channel and 3-channel depth images
        if depth_image.shape[-1] == 3:
            # Take first channel if it's 3-channel (MMFi case)
            depth_image = depth_image[:, :, 0]
        else:
            depth_image = depth_image.squeeze()
        
        # Invert depth if specified (MMFi case where high values = close)
        if denorm_params and denorm_params.get('depth_invert', False):
            depth_image = 255.0 - depth_image
        
        # Normalize depth for better visualization
        depth_nonzero = depth_image[depth_image > 0]
        if len(depth_nonzero) > 0:
            vmin, vmax = np.percentile(depth_nonzero, [2, 98])
            depth_image_vis = np.clip(depth_image, vmin, vmax)
        else:
            depth_image_vis = depth_image
        axes[plot_idx].imshow(depth_image_vis, cmap='jet')
        axes[plot_idx].set_title('Depth Image')
        axes[plot_idx].axis('off')
    plot_idx += 1
    # LiDAR point cloud
    if 'input_lidar' in batch and batch['input_lidar'] is not None:
        lidar_points = batch['input_lidar'][sample_idx][-1].cpu().numpy()
        if 'input_lidar_affine' in batch and batch['input_lidar_affine'] is not None:
            affine_matrix = batch['input_lidar_affine'][sample_idx].cpu().numpy()
            lidar_points = reverse_affine_transform(lidar_points, affine_matrix)
        bounds_lidar = get_bounds(lidar_points)
        plot_3d_point_cloud(axes[plot_idx], lidar_points)
        set_3d_ax_limits(axes[plot_idx], bounds_lidar)
        axes[plot_idx].set_title('LiDAR Point Cloud')
    plot_idx += 1
    # mmWave point cloud
    if 'input_mmwave' in batch and batch['input_mmwave'] is not None:
        mmwave_points = batch['input_mmwave'][sample_idx][-1].cpu().numpy()
        if 'input_mmwave_affine' in batch and batch['input_mmwave_affine'] is not None:
            affine_matrix = batch['input_mmwave_affine'][sample_idx].cpu().numpy()
            mmwave_points = reverse_affine_transform(mmwave_points, affine_matrix)
        bounds_mmwave = get_bounds(mmwave_points)
        plot_3d_point_cloud(axes[plot_idx], mmwave_points)
        set_3d_ax_limits(axes[plot_idx], bounds_mmwave)
        axes[plot_idx].set_title('mmWave Point Cloud')
    plot_idx += 1
    
    gt_keypoints_3d = batch['gt_keypoints'][sample_idx].cpu().numpy()
    pred_keypoints_3d = pred_dict['pred_keypoints'][sample_idx].cpu().numpy()
    bounds = get_bounds(np.concatenate([gt_keypoints_3d, pred_keypoints_3d], axis=0))
    
    # Load GT vertices if available
    gt_vertices = None
    gt_faces = None
    if smpl_model is not None and 'gt_vertices' in batch and batch['gt_vertices'] is not None:
        if isinstance(batch['gt_vertices'], list):
            gt_vertices = batch['gt_vertices'][sample_idx]
        else:
            gt_vertices = batch['gt_vertices'][sample_idx]
        
        if gt_vertices is not None:
            if isinstance(gt_vertices, np.ndarray):
                pass
            elif hasattr(gt_vertices, 'cpu'):
                gt_vertices = gt_vertices.cpu().numpy()
            gt_faces = smpl_model.th_faces.cpu().numpy()
        elif 'gt_smpl' in batch:
            gt_smpl_data = batch['gt_smpl']
            if isinstance(gt_smpl_data, list):
                gt_smpl = gt_smpl_data[sample_idx]
                global_orient = gt_smpl['global_orient']
                body_pose = gt_smpl['body_pose']
                betas = gt_smpl['betas']
                transl = gt_smpl['transl']
            else:
                global_orient = gt_smpl_data['global_orient'][sample_idx].cpu().numpy()
                body_pose = gt_smpl_data['body_pose'][sample_idx].cpu().numpy()
                betas = gt_smpl_data['betas'][sample_idx].cpu().numpy()
                transl = gt_smpl_data['transl'][sample_idx].cpu().numpy()
            
            if hasattr(global_orient, 'cpu'):
                global_orient = global_orient.cpu().numpy()
            if hasattr(body_pose, 'cpu'):
                body_pose = body_pose.cpu().numpy()
            if hasattr(betas, 'cpu'):
                betas = betas.cpu().numpy()
            if hasattr(transl, 'cpu'):
                transl = transl.cpu().numpy()
            
            gt_vertices, gt_faces, _ = get_smpl_mesh_from_params(
                smpl_model, global_orient, body_pose, betas, transl, device=device
            )
    
    # Load Pred vertices if available
    pred_vertices = None
    pred_faces = None
    if smpl_model is not None and 'pred_vertices' in pred_dict and pred_dict['pred_vertices'] is not None:
        if isinstance(pred_dict['pred_vertices'], list):
            pred_vertices = pred_dict['pred_vertices'][sample_idx]
        else:
            pred_vertices = pred_dict['pred_vertices'][sample_idx]
        
        if pred_vertices is not None:
            if isinstance(pred_vertices, np.ndarray):
                pass
            elif hasattr(pred_vertices, 'cpu'):
                pred_vertices = pred_vertices.cpu().numpy()
            pred_faces = smpl_model.th_faces.cpu().numpy()
        elif 'pred_smpl' in pred_dict:
            pred_smpl_data = pred_dict['pred_smpl']
            if isinstance(pred_smpl_data, list):
                pred_smpl = pred_smpl_data[sample_idx]
                global_orient = pred_smpl['global_orient']
                body_pose = pred_smpl['body_pose']
                betas = pred_smpl['betas']
                transl = pred_smpl['transl']
            else:
                global_orient = pred_smpl_data['global_orient'][sample_idx].cpu().numpy()
                body_pose = pred_smpl_data['body_pose'][sample_idx].cpu().numpy()
                betas = pred_smpl_data['betas'][sample_idx].cpu().numpy()
                transl = pred_smpl_data['transl'][sample_idx].cpu().numpy()
            
            if hasattr(global_orient, 'cpu'):
                global_orient = global_orient.cpu().numpy()
            if hasattr(body_pose, 'cpu'):
                body_pose = body_pose.cpu().numpy()
            if hasattr(betas, 'cpu'):
                betas = betas.cpu().numpy()
            if hasattr(transl, 'cpu'):
                transl = transl.cpu().numpy()
            
            pred_vertices, pred_faces, _ = get_smpl_mesh_from_params(
                smpl_model, global_orient, body_pose, betas, transl, device=device
            )
    
    # GT 3D skeleton + mesh overlay
    plot_3d_skeleton(axes[plot_idx], gt_keypoints_3d, edges=bones)
    if gt_vertices is not None and gt_faces is not None:
        plot_smpl_mesh_3d(axes[plot_idx], gt_vertices, gt_faces, alpha=0.3)
        mesh_bounds = get_bounds(gt_vertices)
        set_3d_ax_limits(axes[plot_idx], mesh_bounds)
    else:
        set_3d_ax_limits(axes[plot_idx], bounds)
    axes[plot_idx].set_title('GT 3D Skeleton + Mesh' if gt_vertices is not None else 'GT 3D Skeleton')
    plot_idx += 1
    # GT projections + mesh overlay
    for dims, plane in zip([[0, 1], [0, 2], [1, 2]], ['XY', 'XZ', 'YZ']):
        plot_2d_skeleton(axes[plot_idx], gt_keypoints_3d, dims=dims, edges=bones)
        if gt_vertices is not None:
            vertices_2d_proj = gt_vertices[:, dims]
            axes[plot_idx].scatter(vertices_2d_proj[:, 0], vertices_2d_proj[:, 1], 
                                  color=SMPL_MESH_COLOR, s=0.5, alpha=0.3)
        set_2d_ax_limits(axes[plot_idx], bounds, dims=dims)
        axes[plot_idx].set_title(f'GT {plane} + Mesh' if gt_vertices is not None else f'GT {plane} Projection')
        plot_idx += 1
    # Pred 3D skeleton + mesh overlay + mesh overlay
    plot_3d_skeleton(axes[plot_idx], pred_keypoints_3d, edges=bones)
    if pred_vertices is not None and pred_faces is not None:
        plot_smpl_mesh_3d(axes[plot_idx], pred_vertices, pred_faces, alpha=0.3,
                         color=[0.85, 0.65, 0.65])
        mesh_bounds = get_bounds(pred_vertices)
        set_3d_ax_limits(axes[plot_idx], mesh_bounds)
    else:
        set_3d_ax_limits(axes[plot_idx], bounds)
    axes[plot_idx].set_title('Pred 3D Skeleton + Mesh' if pred_vertices is not None else 'Predicted 3D Skeleton')
    plot_idx += 1
    # Pred projections + mesh overlay + mesh overlay
    for dims, plane in zip([[0, 1], [0, 2], [1, 2]], ['XY', 'XZ', 'YZ']):
        plot_2d_skeleton(axes[plot_idx], pred_keypoints_3d, dims=dims, edges=bones)
        if pred_vertices is not None:
            vertices_2d_proj = pred_vertices[:, dims]
            axes[plot_idx].scatter(vertices_2d_proj[:, 0], vertices_2d_proj[:, 1], 
                                  color=[0.85, 0.65, 0.65], s=0.5, alpha=0.3)
        set_2d_ax_limits(axes[plot_idx], bounds, dims=dims)
        axes[plot_idx].set_title(f'Pred {plane} + Mesh' if pred_vertices is not None else f'Predicted {plane} Projection')
        plot_idx += 1

    # Row 4: Overlay 2D projections on RGB and Depth images
    anchor_key = batch.get('anchor_key', ['input_rgb'])[sample_idx] if isinstance(batch.get('anchor_key', ['input_rgb']), list) else batch.get('anchor_key', 'input_rgb')
    
    # RGB image with GT projected 2D keypoints
    if rgb_image is not None and 'rgb_camera' in batch:
        rgb_camera = batch['rgb_camera'][sample_idx]
        intrinsic = rgb_camera['intrinsic']
        extrinsic = rgb_camera['extrinsic']
        
        if not isinstance(intrinsic, np.ndarray):
            intrinsic = intrinsic.cpu().numpy()
        if not isinstance(extrinsic, np.ndarray):
            extrinsic = extrinsic.cpu().numpy()
        
        # Transform keypoints if anchor is not RGB
        gt_keypoints_for_rgb = gt_keypoints_3d.copy()
        if anchor_key != 'input_rgb':
            # Need to transform from anchor frame to RGB frame
            # Convert 3x4 to 4x4
            extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
            # Transform to RGB camera space
            gt_keypoints_hom = np.hstack([gt_keypoints_for_rgb, np.ones((gt_keypoints_for_rgb.shape[0], 1))])
            gt_keypoints_rgb_hom = (extrinsic_4x4 @ gt_keypoints_hom.T).T
            gt_keypoints_for_rgb = gt_keypoints_rgb_hom[:, :3]
        
        # Project 3D keypoints to 2D
        keypoints_2d = project_3d_to_2d(gt_keypoints_for_rgb, intrinsic)
        
        # Overlay mesh if available
        if gt_vertices is not None and gt_faces is not None:
            gt_vertices_for_rgb = gt_vertices.copy()
            if anchor_key != 'input_rgb':
                extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
                gt_verts_hom = np.hstack([gt_vertices_for_rgb, np.ones((gt_vertices_for_rgb.shape[0], 1))])
                gt_verts_rgb_hom = (extrinsic_4x4 @ gt_verts_hom.T).T
                gt_vertices_for_rgb = gt_verts_rgb_hom[:, :3]
            
            plot_smpl_mesh_on_image(axes[plot_idx], rgb_image.copy(), gt_vertices_for_rgb, 
                                   gt_faces, intrinsic, alpha=0.5, edge_color=None)
        else:
            axes[plot_idx].imshow(rgb_image.copy())
        
        # Plot RGB with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], None, keypoints_2d, edges=bones)
        axes[plot_idx].set_title('RGB + GT Overlay + Mesh' if gt_vertices is not None else 'RGB + GT 2D Overlay')
    plot_idx += 1
    
    # Depth image with GT projected 2D keypoints
    if depth_image is not None and 'depth_camera' in batch:
        depth_camera = batch['depth_camera'][sample_idx]
        intrinsic = depth_camera['intrinsic']
        extrinsic = depth_camera['extrinsic']
        
        if not isinstance(intrinsic, np.ndarray):
            intrinsic = intrinsic.cpu().numpy()
        if not isinstance(extrinsic, np.ndarray):
            extrinsic = extrinsic.cpu().numpy()
        
        # Transform keypoints if anchor is not depth
        gt_keypoints_for_depth = gt_keypoints_3d.copy()
        if anchor_key != 'input_depth':
            # Need to transform from anchor frame to depth frame
            # Convert 3x4 to 4x4
            extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
            # Transform to depth camera space
            gt_keypoints_hom = np.hstack([gt_keypoints_for_depth, np.ones((gt_keypoints_for_depth.shape[0], 1))])
            gt_keypoints_depth_hom = (extrinsic_4x4 @ gt_keypoints_hom.T).T
            gt_keypoints_for_depth = gt_keypoints_depth_hom[:, :3]
        
        # Project 3D keypoints to 2D
        keypoints_2d = project_3d_to_2d(gt_keypoints_for_depth, intrinsic)
        
        # Handle both single-channel and 3-channel depth images
        depth_for_overlay = depth_image.copy()
        if depth_for_overlay.ndim == 3 and depth_for_overlay.shape[-1] == 3:
            depth_for_overlay = depth_for_overlay[:, :, 0]
        
        # Convert grayscale to RGB for overlay with better contrast
        depth_nonzero = depth_for_overlay[depth_for_overlay > 0]
        if len(depth_nonzero) > 0:
            vmin, vmax = np.percentile(depth_nonzero, [2, 98])
            depth_image_norm = np.clip((depth_for_overlay - vmin) / (vmax - vmin + 1e-6) * 255, 0, 255).astype(np.uint8)
        else:
            depth_image_norm = depth_for_overlay.astype(np.uint8)
        depth_image_rgb = np.stack([depth_image_norm, depth_image_norm, depth_image_norm], axis=-1)
        
        # Overlay mesh if available
        if gt_vertices is not None and gt_faces is not None:
            gt_vertices_for_depth = gt_vertices.copy()
            if anchor_key != 'input_depth':
                extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
                gt_verts_hom = np.hstack([gt_vertices_for_depth, np.ones((gt_vertices_for_depth.shape[0], 1))])
                gt_verts_depth_hom = (extrinsic_4x4 @ gt_verts_hom.T).T
                gt_vertices_for_depth = gt_verts_depth_hom[:, :3]
            
            plot_smpl_mesh_on_image(axes[plot_idx], depth_image_rgb, gt_vertices_for_depth, 
                                   gt_faces, intrinsic, alpha=0.5, edge_color=None)
        else:
            axes[plot_idx].imshow(depth_image_rgb)
        
        # Plot depth with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], None, keypoints_2d, edges=bones)
        axes[plot_idx].set_title('Depth + GT Overlay + Mesh' if gt_vertices is not None else 'Depth + GT 2D Overlay')
    plot_idx += 1
    
    # RGB image with Pred projected 2D keypoints
    if rgb_image is not None and 'rgb_camera' in batch:
        rgb_camera = batch['rgb_camera'][sample_idx]
        intrinsic = rgb_camera['intrinsic']
        extrinsic = rgb_camera['extrinsic']
        
        if not isinstance(intrinsic, np.ndarray):
            intrinsic = intrinsic.cpu().numpy()
        if not isinstance(extrinsic, np.ndarray):
            extrinsic = extrinsic.cpu().numpy()
        
        # Transform keypoints if anchor is not RGB
        pred_keypoints_for_rgb = pred_keypoints_3d.copy()
        if anchor_key != 'input_rgb':
            # Need to transform from anchor frame to RGB frame
            extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
            pred_keypoints_hom = np.hstack([pred_keypoints_for_rgb, np.ones((pred_keypoints_for_rgb.shape[0], 1))])
            pred_keypoints_rgb_hom = (extrinsic_4x4 @ pred_keypoints_hom.T).T
            pred_keypoints_for_rgb = pred_keypoints_rgb_hom[:, :3]
        
        # Project 3D keypoints to 2D
        keypoints_2d = project_3d_to_2d(pred_keypoints_for_rgb, intrinsic)
        
        # Overlay mesh if available
        if pred_vertices is not None and pred_faces is not None:
            pred_vertices_for_rgb = pred_vertices.copy()
            if anchor_key != 'input_rgb':
                extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
                pred_verts_hom = np.hstack([pred_vertices_for_rgb, np.ones((pred_vertices_for_rgb.shape[0], 1))])
                pred_verts_rgb_hom = (extrinsic_4x4 @ pred_verts_hom.T).T
                pred_vertices_for_rgb = pred_verts_rgb_hom[:, :3]
            
            plot_smpl_mesh_on_image(axes[plot_idx], rgb_image.copy(), pred_vertices_for_rgb, 
                                   pred_faces, intrinsic, color=[0.85, 0.65, 0.65], alpha=0.5, edge_color=None)
        else:
            axes[plot_idx].imshow(rgb_image.copy())
        
        # Plot RGB with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], None, keypoints_2d, edges=bones)
        axes[plot_idx].set_title('RGB + Pred Overlay + Mesh' if pred_vertices is not None else 'RGB + Pred 2D Overlay')
    plot_idx += 1
    
    # Depth image with Pred projected 2D keypoints
    if depth_image is not None and 'depth_camera' in batch:
        depth_camera = batch['depth_camera'][sample_idx]
        intrinsic = depth_camera['intrinsic']
        extrinsic = depth_camera['extrinsic']
        
        if not isinstance(intrinsic, np.ndarray):
            intrinsic = intrinsic.cpu().numpy()
        if not isinstance(extrinsic, np.ndarray):
            extrinsic = extrinsic.cpu().numpy()
        
        # Transform keypoints if anchor is not depth
        pred_keypoints_for_depth = pred_keypoints_3d.copy()
        if anchor_key != 'input_depth':
            # Need to transform from anchor frame to depth frame
            extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
            pred_keypoints_hom = np.hstack([pred_keypoints_for_depth, np.ones((pred_keypoints_for_depth.shape[0], 1))])
            pred_keypoints_depth_hom = (extrinsic_4x4 @ pred_keypoints_hom.T).T
            pred_keypoints_for_depth = pred_keypoints_depth_hom[:, :3]
        
        # Project 3D keypoints to 2D
        keypoints_2d = project_3d_to_2d(pred_keypoints_for_depth, intrinsic)
        
        # Handle both single-channel and 3-channel depth images
        depth_for_overlay = depth_image.copy()
        if depth_for_overlay.ndim == 3 and depth_for_overlay.shape[-1] == 3:
            depth_for_overlay = depth_for_overlay[:, :, 0]
        
        # Convert grayscale to RGB for overlay with better contrast
        depth_nonzero = depth_for_overlay[depth_for_overlay > 0]
        if len(depth_nonzero) > 0:
            vmin, vmax = np.percentile(depth_nonzero, [2, 98])
            depth_image_norm = np.clip((depth_for_overlay - vmin) / (vmax - vmin + 1e-6) * 255, 0, 255).astype(np.uint8)
        else:
            depth_image_norm = depth_for_overlay.astype(np.uint8)
        depth_image_rgb = np.stack([depth_image_norm, depth_image_norm, depth_image_norm], axis=-1)
        
        # Overlay mesh if available
        if pred_vertices is not None and pred_faces is not None:
            pred_vertices_for_depth = pred_vertices.copy()
            if anchor_key != 'input_depth':
                extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
                pred_verts_hom = np.hstack([pred_vertices_for_depth, np.ones((pred_vertices_for_depth.shape[0], 1))])
                pred_verts_depth_hom = (extrinsic_4x4 @ pred_verts_hom.T).T
                pred_vertices_for_depth = pred_verts_depth_hom[:, :3]
            
            plot_smpl_mesh_on_image(axes[plot_idx], depth_image_rgb, pred_vertices_for_depth, 
                                   pred_faces, intrinsic, color=[0.85, 0.65, 0.65], alpha=0.5, edge_color=None)
        else:
            axes[plot_idx].imshow(depth_image_rgb)
        
        # Plot depth with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], None, keypoints_2d, edges=bones)
        axes[plot_idx].set_title('Depth + Pred Overlay + Mesh' if pred_vertices is not None else 'Depth + Pred 2D Overlay')
    plot_idx += 1

    plt.tight_layout()
    return fig
    