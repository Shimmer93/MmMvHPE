import matplotlib.pyplot as plt
import numpy as np
from .skeleton import JOINT_COLOR_MAP

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
    
    min_vals = bounds_arr[::2][dims]
    max_vals = bounds_arr[1::2][dims]
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

def plot_2d_point_cloud(ax, points, dims=[0, 1], color='k', s=1):
    ax.scatter(points[:, dims[0]], points[:, dims[1]], c=color, s=s)

def plot_3d_point_cloud(ax, points, color='k', s=1):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=s)

def visualize_multimodal_sample(batch, pred_dict, denorm_params):
    # Plot the following subplots:
    # 1. RGB image (if available)
    # 2. Depth image (if available)
    # 3. LiDAR point cloud (if available)
    # 4. mmWave point cloud (if available)
    # 5. GT 3D skeleton
    # 6-8. GT 3D skeleton projected onto XY, XZ, YZ planes
    # 9. Predicted 3D skeleton
    # 10-12. Predicted 3D skeleton projected onto XY, XZ, YZ planes
    
    # first row: RGB, Depth, LiDAR, mmWave (if not available, leave blank)
    # second row: GT 3D skeleton, GT XY, GT XZ, GT YZ
    # third row: Pred 3D skeleton, Pred XY, Pred XZ, Pred YZ

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    sample_idx = 0  # visualize the first sample in the batch
    plot_idx = 0

    # RGB image
    if 'input_rgb' in batch and batch['input_rgb'] is not None:
        rgb_image = batch['input_rgb'][sample_idx].permute(1, 2, 0).cpu().numpy()
        rgb_image = denormalize(rgb_image, denorm_params['rgb_mean'], denorm_params['rgb_std'])
        axes[plot_idx].imshow(rgb_image)
        axes[plot_idx].set_title('RGB Image')
    plot_idx += 1
    # Depth image
    if 'input_depth' in batch and batch['input_depth'] is not None:
        depth_image = batch['input_depth'][sample_idx].squeeze().cpu().numpy()
        depth_image = denormalize(depth_image, denorm_params['depth_mean'], denorm_params['depth_std'])
        axes[plot_idx].imshow(depth_image, cmap='gray')
        axes[plot_idx].set_title('Depth Image')
    plot_idx += 1
    # LiDAR point cloud
    if 'input_lidar' in batch and batch['input_lidar'] is not None:
        lidar_points = batch['input_lidar'][sample_idx].cpu().numpy()
        if 'input_lidar_affine' in batch and batch['input_lidar_affine'] is not None:
            affine_matrix = batch['input_lidar_affine'][sample_idx].cpu().numpy()
            lidar_points = reverse_affine_transform(lidar_points, affine_matrix)
        plot_3d_point_cloud(axes[plot_idx], lidar_points)
        axes[plot_idx].set_title('LiDAR Point Cloud')
    plot_idx += 1
    # mmWave point cloud
    if 'input_mmwave' in batch and batch['input_mmwave'] is not None:
        mmwave_points = batch['input_mmwave'][sample_idx].cpu().numpy()
        if 'input_mmwave_affine' in batch and batch['input_mmwave_affine'] is not None:
            affine_matrix = batch['input_mmwave_affine'][sample_idx].cpu().numpy()
            mmwave_points = reverse_affine_transform(mmwave_points, affine_matrix)
        plot_3d_point_cloud(axes[plot_idx], mmwave_points)
        axes[plot_idx].set_title('mmWave Point Cloud')
    plot_idx += 1
    
    gt_keypoints_3d = batch['gt_keypoints'][sample_idx].cpu().numpy()
    pred_keypoints_3d = pred_dict['pred_keypoints'][sample_idx].cpu().numpy()
    bounds = get_bounds(np.concatenate([gt_keypoints_3d, pred_keypoints_3d], axis=0))
    
    # GT 3D skeleton
    plot_3d_skeleton(axes[plot_idx], gt_keypoints_3d)
    set_3d_ax_limits(axes[plot_idx], bounds)
    axes[plot_idx].set_title('GT 3D Skeleton')
    plot_idx += 1
    # GT projections
    for dims, plane in zip([[0, 1], [0, 2], [1, 2]], ['XY', 'XZ', 'YZ']):
        plot_2d_skeleton(axes[plot_idx], gt_keypoints_3d, dims=dims)
        set_2d_ax_limits(axes[plot_idx], bounds, dims=dims)
        axes[plot_idx].set_title(f'GT {plane} Projection')
        plot_idx += 1
    # Pred 3D skeleton
    plot_3d_skeleton(axes[plot_idx], pred_keypoints_3d)
    set_3d_ax_limits(axes[plot_idx], bounds)
    axes[plot_idx].set_title('Predicted 3D Skeleton')
    plot_idx += 1
    # Pred projections
    for dims, plane in zip([[0, 1], [0, 2], [1, 2]], ['XY', 'XZ', 'YZ']):
        plot_2d_skeleton(axes[plot_idx], pred_keypoints_3d, dims=dims)
        set_2d_ax_limits(axes[plot_idx], bounds, dims=dims)
        axes[plot_idx].set_title(f'Predicted {plane} Projection')
        plot_idx += 1

    plt.tight_layout()
    return fig
    