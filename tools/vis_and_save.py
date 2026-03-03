'''
Visualize the inputs and outputs of the model, and save the results to disk.
Input:
    - rgb image: (H, W, 3), H=W=224 in our case
    - pc: (N, 3)
    - camera poses: if you can find the way to visualize the camera poses, please discuss with me first. if you cannot, we can skip this part for now.
Output:
    - the keys can be found in models/modal_api.py L751-759
    - the global keypoints going through tools/eval_fixed_lidar_frame.py using
        - pred_cameras_stream
        - pred_keypoints
    - the mesh generated from pred_smpl_params (refer to tools/vis_smpl.py)
Note:
    - Input can be copied from /opt/data/humman_cropped
    - It will be better if you make the ratio of output images identical
    - Make sure the aspect ratio of 3D visualizations is correct, plt is not good at this
    - You don't need to visualize all data, you can specify the number of samples to visualize, e.g., 100
    - The predicted keypoints from tools/eval_fixed_lidar_frame.py are in a sensor coordinate system, not the original world system. To make the visualization better-looking, you can transform the predicted keypoints back to the original world coordinate system using the original camera poses. 

'''

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import SMPL and utils based on your project structure
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models.smpl import SMPL
from misc.utils import load
from vis_smpl import id_to_file_name_humman, get_input_data

# NEW IMPORTS FOR 2D PROJECTION ONLY
from misc.pose_enc import pose_encoding_to_extri_intri
from eval_fixed_lidar_frame import (
    _get_camera_index, 
    _extract_camera_encoding, 
    _get_sample_keypoints,
    _pose_encoding_to_extrinsic,
    _get_sample_lidar_center,
    _inverse_lidar_camera_center
)

# --- Define SMPL Skeleton Connections (Bones) ---
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)
]

def draw_skeleton_3d(ax, keypoints, color='red', linewidth=2):
    """Draws lines connecting the 3D keypoints."""
    for (i, j) in SMPL_BONES:
        if i < len(keypoints) and j < len(keypoints):
            ax.plot([keypoints[i, 0], keypoints[j, 0]],
                    [keypoints[i, 1], keypoints[j, 1]],
                    [keypoints[i, 2], keypoints[j, 2]], color=color, linewidth=linewidth)

def draw_skeleton_2d(ax, keypoints_2d, color='red', linewidth=2):
    """Draws lines connecting the 2D keypoints on the image."""
    for (i, j) in SMPL_BONES:
        if i < len(keypoints_2d) and j < len(keypoints_2d):
            ax.plot([keypoints_2d[i, 0], keypoints_2d[j, 0]],
                    [keypoints_2d[i, 1], keypoints_2d[j, 1]], color=color, linewidth=linewidth)

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_box_aspect([1, 1, 1])

def get_smpl_vertices(smpl_model, pose, global_orient, beta, translation):
    smpl_model.eval()
    with torch.no_grad():
        pose_tensor = torch.from_numpy(pose).float().unsqueeze(0)
        global_orient_tensor = torch.from_numpy(global_orient).float().unsqueeze(0)
        full_pose = torch.cat([global_orient_tensor, pose_tensor], dim=1)
        beta_tensor = torch.from_numpy(beta).float().unsqueeze(0)
        translation_tensor = torch.from_numpy(translation).float().unsqueeze(0)

        verts, _ = smpl_model(
            th_pose_axisang=full_pose,
            th_betas=beta_tensor,
            th_trans=translation_tensor,
        )
    return verts[0].cpu().numpy()

def visualize_and_save(pred_file, data_root="/opt/data/humman_cropped", save_dir="visualization_results", num_samples=3):
    os.makedirs(save_dir, exist_ok=True)
    preds = load(pred_file)
    sample_ids = preds['sample_ids']
    num_to_vis = min(num_samples, len(sample_ids))
    smpl_model = SMPL(model_path='/opt/data/SMPL_NEUTRAL.pkl')
    faces = smpl_model.th_faces.cpu().numpy()
    
    for i in range(num_to_vis):
        sample_id = sample_ids[i]
        fig = plt.figure(figsize=(18, 12))
        
        rgb_image, _, pc, _ = get_input_data(sample_id, dataset='humman_preproc', data_root=data_root)
        
        keypoints3d = preds['pred_smpl_keypoints'][i]
        
        smpl_params = preds['pred_smpl_params'][i]
        pose = smpl_params[3:72]
        global_orient = np.zeros(3)
        beta = smpl_params[72:82]
        translation = np.zeros(3)
        
        vertices3d = get_smpl_vertices(smpl_model, pose, global_orient, beta, translation)

        # --- Strict Camera-Based 2D Projection Mapping ---
        # Extract the RGB camera exactly like eval_fixed_lidar_frame
        try:
            rgb_idx = _get_camera_index(preds, i, modality="rgb", fallback_idx=0, use_stream_index=True, sensor_idx=0)
            rgb_cam_enc = _extract_camera_encoding(preds['pred_cameras_stream'], i, rgb_idx)
            
            # 1. Extract Extrinsic (R and T) to map Camera <-> World
            rgb_extr = _pose_encoding_to_extrinsic(rgb_cam_enc, "absT_quaR_FoV")
            R = rgb_extr[:, :3]
            T = rgb_extr[:, 3]
            
            # 2. Extract Intrinsic (K)
            pe_tensor = torch.tensor(rgb_cam_enc, dtype=torch.float32).view(1, 1, -1)
            _, intri = pose_encoding_to_extri_intri(pe_tensor, image_size_hw=(224, 224), pose_encoding_type="absT_quaR_FoV", build_intrinsics=True)
            K = intri[0, 0].numpy()
            
            # 3. Transform WORLD keypoints (keypoints3d) backwards into the RGB CAMERA frame
            # The extrinsic projects Camera -> World via (X @ R^T + T)
            # Inverse projection World -> Camera is (X - T) @ R
            cam_kps = (keypoints3d - T) @ R
            
            # 4. Project Local 3D Camera Keypoints -> 2D Image Pixels
            kps_2d_homo = cam_kps @ K.T
            u = kps_2d_homo[:, 0] / kps_2d_homo[:, 2]
            v = kps_2d_homo[:, 1] / kps_2d_homo[:, 2]
            kps_2d = np.stack([u, v], axis=-1)
            
            # --- NEW: Convert SMPL World Skeleton to LiDAR frame for ax2 ---
            
            # 1. Get LiDAR camera params (LiDAR -> World Extrinsic)
            lidar_idx = _get_camera_index(preds, i, modality="lidar", fallback_idx=0, use_stream_index=True, sensor_idx=0)
            lidar_cam_enc = _extract_camera_encoding(preds['pred_cameras_stream'], i, lidar_idx)
            lidar_center = _get_sample_lidar_center(preds, i)
            lidar_cam_enc = _inverse_lidar_camera_center(lidar_cam_enc, lidar_center)
            lidar_extr = _pose_encoding_to_extrinsic(lidar_cam_enc, "absT_quaR_FoV")
            
            R_lidar = lidar_extr[:, :3]
            T_lidar = lidar_extr[:, 3]
            
            # 2. World Frame -> LiDAR Sensor Frame
            # Utilizing the structurally perfect SMPL keypoints directly
            lidar_kps = (keypoints3d - T_lidar) @ R_lidar
            
        except Exception as e:
            print(f"Skipping 2D projection for sample {i}: {e}")
            kps_2d = np.zeros((keypoints3d.shape[0], 2))
            lidar_kps = keypoints3d # fallback

        # --- Subplot 1: RGB + 2D Skeleton ---
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(rgb_image)
        ax1.scatter(kps_2d[:, 0], kps_2d[:, 1], s=20, c='red', zorder=3)
        draw_skeleton_2d(ax1, kps_2d, color='cyan', linewidth=2)  # Connected 2D Joints
        ax1.set_title('RGB Image & Skeleton')
        ax1.axis('off')
        
        # --- Subplot 2: Point Cloud & 3D Skeleton Overlap ---
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='gray', alpha=0.5, label='Point Cloud')
        # Use exclusively the mapped LiDAR keypoints here
        ax2.scatter(lidar_kps[:, 0], lidar_kps[:, 1], lidar_kps[:, 2], s=20, c='red', label='Keypoints')
        draw_skeleton_3d(ax2, lidar_kps, color='red')  # Connected 3D Joints
        ax2.set_title('Absolute World Overlap')
        set_axes_equal(ax2)
        
        # Wrapper for SMPL mesh + skeleton drawing
        mins = vertices3d.min(axis=0)
        maxs = vertices3d.max(axis=0)
        def draw_mesh(ax, elev, azim, title):
            mesh = Poly3DCollection(vertices3d[faces], alpha=0.5, facecolor=[0.65, 0.75, 0.85])
            ax.add_collection3d(mesh)
            ax.scatter(keypoints3d[:, 0], keypoints3d[:, 1], keypoints3d[:, 2], s=15, c='red')
            draw_skeleton_3d(ax, keypoints3d, color='red')  # Connected 3D Joints
            
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            set_axes_equal(ax)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title)

        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        draw_mesh(ax3, elev=30, azim=-60, title='SMPL (Perspective)')

        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        draw_mesh(ax4, elev=90, azim=-90, title='SMPL XY View (Top)')

        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        draw_mesh(ax5, elev=0, azim=0, title='SMPL YZ View (Side)')

        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        draw_mesh(ax6, elev=0, azim=-90, title='SMPL ZX View (Front)')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{sample_id}.png')
        plt.savefig(save_path)
        plt.close(fig)
        
    print(f"Saved {num_to_vis} visualizations to {save_dir}")

if __name__ == "__main__":
    # Example usage
    visualize_and_save("logs/fasternet/0/HummanVIBEToken_test_predictions.pkl")