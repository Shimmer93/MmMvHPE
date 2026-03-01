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