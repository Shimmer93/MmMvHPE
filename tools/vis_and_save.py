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
from eval_fixed_lidar_frame import transform_points_to_world

# Import SMPL and utils based on your project structure
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models.smpl import SMPL
from misc.utils import load
from vis_smpl import id_to_file_name_humman, get_input_data
# Import the transformation functions from your evaluation script
from eval_fixed_lidar_frame import _pose_encoding_to_extrinsic, _transform_points


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
    
    # Using the exact data ranges for the box aspect prevents stretching 
    # and keeps the bounding box rectangular, avoiding unnecessary cubic padding.
    # ax.set_box_aspect([x_range, y_range, z_range])

def transform_to_world(points, camera_pose):
    """
    Transform points from sensor coordinates to world coordinates.
    Assuming camera_pose contains [R | T] or similar encoding.
    You will need to adjust this based on your exact camera_pose format.
    """
    # camera_pose is a 9D encoding (absT_quaR_FoV)
    if camera_pose is None or not np.isfinite(camera_pose).all():
        return points

    # Convert the 9D pose encoding to a 3x4 extrinsic matrix [R | T]
    extrinsic = _pose_encoding_to_extrinsic(camera_pose, pose_encoding_type="absT_quaR_FoV")
    
    # Apply R.T @ points + T
    world_points = _transform_points(points, extrinsic)
    
    return world_points

def get_smpl_vertices(smpl_model, smpl_params):
    """Extract vertices from pred_smpl_params."""
    # Based on vis_smpl.py L60-68
    pose = smpl_params[3:72]
    global_orient = np.zeros(3) # Or extract if available
    beta = smpl_params[72:82]
    translation = np.zeros(3)   # Or extract if available
    
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

def visualize_and_save(pred_file, data_root="/opt/data/humman_cropped", save_dir="visualization_results", num_samples=100):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load predictions
    preds = load(pred_file)
    sample_ids = preds['sample_ids']
    num_to_vis = min(num_samples, len(sample_ids))
    
    # Load SMPL model
    smpl_model = SMPL(model_path='/opt/data/SMPL_NEUTRAL.pkl')
    faces = smpl_model.th_faces.cpu().numpy()
    
    for i in range(num_to_vis):
        sample_id = sample_ids[i]
        fig = plt.figure(figsize=(18, 6))
        
        # 2. Load Raw Input Data (You can reuse id_to_file_name_humman from vis_smpl.py)
        rgb_image, _, pc, _ = get_input_data(sample_id, dataset='humman_preproc', data_root=data_root)
        # rgb_image = np.zeros((224, 224, 3)) # Placeholder
        # pc = np.random.rand(1000, 3)        # Placeholder
        
        # 3. Transform Keypoints
        raw_keypoints = preds['pred_keypoints'][i]
        camera_pose = preds['pred_cameras_stream'][i]
        print("[DEBUG] camera_pose shape:", camera_pose.shape)
        keypoints_world = transform_to_world(raw_keypoints, camera_pose)
        # keypoints_world = raw_keypoints # Placeholder
        
        # 4. Generate SMPL Mesh
        smpl_params = preds['pred_smpl_params'][i]
        vertices = get_smpl_vertices(smpl_model, smpl_params)
        
        # --- Plotting ---
        # RGB Image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(rgb_image)
        ax1.set_title('RGB Image')
        ax1.axis('off')
        
        # Point Cloud & Keypoints
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='gray', alpha=0.5, label='Point Cloud')
        ax2.scatter(keypoints_world[:, 0], keypoints_world[:, 1], keypoints_world[:, 2], s=20, c='red', label='Keypoints')
        ax2.set_title('Point Cloud & Keypoints')
        set_axes_equal(ax2)
        
        # SMPL Mesh
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        mesh = Poly3DCollection(vertices[faces], alpha=0.5, facecolor=[0.65, 0.75, 0.85])
        ax3.add_collection3d(mesh)
        ax3.scatter(keypoints_world[:, 0], keypoints_world[:, 1], keypoints_world[:, 2], s=20, c='red')
        
        # Set limits for mesh plot based on vertices
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ax3.set_xlim(mins[0], maxs[0])
        ax3.set_ylim(mins[1], maxs[1])
        ax3.set_zlim(mins[2], maxs[2])
        ax3.set_title('SMPL Mesh')
        set_axes_equal(ax3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{sample_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Saved {num_to_vis} visualizations to {save_dir}")

if __name__ == "__main__":
    # Example usage
    visualize_and_save("logs/fasternet/0/HummanVIBEToken_test_predictions.pkl")
    pass