from __future__ import annotations

import numpy as np
import rerun as rr

from .geometry import rotate_points_y


def build_bone_segments(keypoints: np.ndarray, bones: list[tuple[int, int]]) -> list[list[np.ndarray]]:
    """Build line segments for valid bone index pairs."""
    segments = []
    for i, j in bones:
        if 0 <= i < len(keypoints) and 0 <= j < len(keypoints):
            segments.append([keypoints[i], keypoints[j]])
    return segments


def log_point_cloud_3d(
    path: str,
    points: np.ndarray,
    colors=None,
    radii: float = 0.01,
) -> None:
    if points is None or len(points) == 0:
        return

    xyz = points[:, :3]
    if colors is None:
        if points.shape[1] > 3:
            intensity = points[:, 3]
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            colors = np.stack(
                [intensity_norm * 0.3, intensity_norm * 0.7 + 0.3, 1.0 - intensity_norm * 0.5],
                axis=1,
            )
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = [180, 180, 180]

    rr.log(
        path,
        rr.Points3D(
            xyz,
            colors=colors,
            radii=radii if isinstance(radii, (list, np.ndarray)) else [radii] * len(xyz),
        ),
    )


def log_skeleton_3d(
    path: str,
    keypoints: np.ndarray,
    skeleton_class,
    color,
    radius: float = 0.015,
    bones: list[tuple[int, int]] | None = None,
    bone_colors=None,
) -> None:
    rr.log(
        f"{path}/joints",
        rr.Points3D(keypoints, colors=[color] * len(keypoints), radii=[radius] * len(keypoints)),
    )

    if bones is None and hasattr(skeleton_class, "bones") and skeleton_class.bones:
        bones = skeleton_class.bones

    if bones:
        bone_points = build_bone_segments(keypoints, bones)
        if bone_points:
            line_colors = bone_colors if bone_colors is not None else [color] * len(bone_points)
            rr.log(
                f"{path}/bones",
                rr.LineStrips3D(
                    bone_points,
                    colors=line_colors,
                    radii=[radius * 0.5] * len(bone_points),
                ),
            )


def log_mesh_3d(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color=(150, 180, 220),
    alpha: float = 0.3,
) -> None:
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
            albedo_factor=[color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha],
        ),
    )


def log_smpl_views(
    prefix: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    skeleton_class,
    mesh_color,
    mesh_alpha: float,
    skeleton_color,
    skeleton_radius: float,
    log_skeleton: bool = True,
    side_yaw_rad: float = np.deg2rad(90.0),
) -> None:
    log_mesh_3d(f"world/front/{prefix}/mesh", vertices, faces, color=mesh_color, alpha=mesh_alpha)
    if log_skeleton:
        log_skeleton_3d(
            f"world/front/{prefix}/skeleton",
            joints,
            skeleton_class,
            color=skeleton_color,
            radius=skeleton_radius,
        )

    vertices_side = rotate_points_y(vertices, side_yaw_rad)
    joints_side = rotate_points_y(joints, side_yaw_rad)
    log_mesh_3d(f"world/side/{prefix}/mesh", vertices_side, faces, color=mesh_color, alpha=mesh_alpha)
    if log_skeleton:
        log_skeleton_3d(
            f"world/side/{prefix}/skeleton",
            joints_side,
            skeleton_class,
            color=skeleton_color,
            radius=skeleton_radius,
        )


def log_skeleton_views(
    prefix: str,
    keypoints: np.ndarray,
    skeleton_class,
    color,
    radius: float,
    side_yaw_rad: float = np.deg2rad(90.0),
    bones: list[tuple[int, int]] | None = None,
    bone_colors=None,
    side_transform=None,
) -> None:
    log_skeleton_3d(
        f"world/front/{prefix}/skeleton",
        keypoints,
        skeleton_class,
        color=color,
        radius=radius,
        bones=bones,
        bone_colors=bone_colors,
    )
    if side_transform is None:
        keypoints_side = rotate_points_y(keypoints, side_yaw_rad)
    else:
        keypoints_side = side_transform(keypoints)
    log_skeleton_3d(
        f"world/side/{prefix}/skeleton",
        keypoints_side,
        skeleton_class,
        color=color,
        radius=radius,
        bones=bones,
        bone_colors=bone_colors,
    )
