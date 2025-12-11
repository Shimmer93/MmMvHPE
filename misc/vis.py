import matplotlib.pyplot as plt
import numpy as np
from .skeleton import JOINT_COLOR_MAP, H36MSkeleton, COCOSkeleton, SMPLSkeleton

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
        image: RGB image array (H, W, 3)
        keypoints_2d: 2D keypoint coordinates (N, 2) in image space
        edges: list of (i, j) tuples representing skeleton bones
        color_map: color map for joints
        linewidth: line width for bones
        s: marker size for joints
    """
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


def plot_2d_point_cloud(ax, points, dims=[0, 1], color='k', s=1):
    points_ = points[..., :3].reshape(-1, 3)
    ax.scatter(points_[:, dims[0]], points_[:, dims[1]], c=color, s=s)

def plot_3d_point_cloud(ax, points, color='k', s=1):
    points_ = points[..., :3].reshape(-1, 3)
    ax.scatter(points_[:, 0], points_[:, 1], points_[:, 2], c=color, s=s)

def visualize_multimodal_sample(batch, pred_dict, skl_format=None, denorm_params=None):
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
    
    # GT 3D skeleton
    plot_3d_skeleton(axes[plot_idx], gt_keypoints_3d, edges=bones)
    set_3d_ax_limits(axes[plot_idx], bounds)
    axes[plot_idx].set_title('GT 3D Skeleton')
    plot_idx += 1
    # GT projections
    for dims, plane in zip([[0, 1], [0, 2], [1, 2]], ['XY', 'XZ', 'YZ']):
        plot_2d_skeleton(axes[plot_idx], gt_keypoints_3d, dims=dims, edges=bones)
        set_2d_ax_limits(axes[plot_idx], bounds, dims=dims)
        axes[plot_idx].set_title(f'GT {plane} Projection')
        plot_idx += 1
    # Pred 3D skeleton
    plot_3d_skeleton(axes[plot_idx], pred_keypoints_3d, edges=bones)
    set_3d_ax_limits(axes[plot_idx], bounds)
    axes[plot_idx].set_title('Predicted 3D Skeleton')
    plot_idx += 1
    # Pred projections
    for dims, plane in zip([[0, 1], [0, 2], [1, 2]], ['XY', 'XZ', 'YZ']):
        plot_2d_skeleton(axes[plot_idx], pred_keypoints_3d, dims=dims, edges=bones)
        set_2d_ax_limits(axes[plot_idx], bounds, dims=dims)
        axes[plot_idx].set_title(f'Predicted {plane} Projection')
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
        
        # Plot RGB with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], rgb_image.copy(), keypoints_2d, edges=bones)
        axes[plot_idx].set_title('RGB + GT 2D Overlay')
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
        
        # Plot depth with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], depth_image_rgb, keypoints_2d, edges=bones)
        axes[plot_idx].set_title('Depth + GT 2D Overlay')
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
        
        # Plot RGB with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], rgb_image.copy(), keypoints_2d, edges=bones)
        axes[plot_idx].set_title('RGB + Pred 2D Overlay')
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
        
        # Plot depth with 2D skeleton overlay
        plot_2d_skeleton_on_image(axes[plot_idx], depth_image_rgb, keypoints_2d, edges=bones)
        axes[plot_idx].set_title('Depth + Pred 2D Overlay')
    plot_idx += 1

    plt.tight_layout()
    return fig
    