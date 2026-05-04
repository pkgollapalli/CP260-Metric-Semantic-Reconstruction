"""Visualization: matplotlib (always works) + Open3D (if available)."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _camera_frustum_lines(K, R, t, scale=0.05, W=640, H=480):
    """Return 5 frustum corners in world coords: cam center + 4 image-plane corners."""
    K_inv = np.linalg.inv(K)
    corners_px = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]],
                          dtype=np.float64)
    rays_cam = (K_inv @ corners_px.T).T  # (4,3)
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
    pts_cam = rays_cam * scale  # at distance `scale`
    # cam center in world
    C = -R.T @ t
    # rotate cam-frame points into world: world = R^T @ cam + C
    pts_world = (R.T @ pts_cam.T).T + C
    return C, pts_world  # (3,), (4,3)


def plot_scene_3d(points, colors, frames, save_path=None,
                  obj_poses=None, frustum_scale=0.08):
    """3D plot of points + camera frustums + object boxes."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # downsample for speed
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        P = points[idx]; C = colors[idx]
    else:
        P, C = points, colors

    if len(P):
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.clip(C, 0, 1), s=0.5, alpha=0.6)

    # frustums
    for f in frames:
        H, W = f["image"].shape[:2]
        Cw, corners = _camera_frustum_lines(f["K"], f["R"], f["t"],
                                            scale=frustum_scale, W=W, H=H)
        # lines from cam center to 4 corners
        for i in range(4):
            ax.plot([Cw[0], corners[i, 0]], [Cw[1], corners[i, 1]],
                    [Cw[2], corners[i, 2]], 'r-', lw=0.7, alpha=0.7)
        # rectangle
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]], 'r-', lw=0.7, alpha=0.7)

    # object oriented boxes
    if obj_poses:
        for op in obj_poses:
            if op.get("status") != "ok":
                continue
            c = np.array(op["center"]); ext = np.array(op["extent"])
            R = np.array(op["rotation_matrix"])
            # 8 corners
            signs = np.array([[s1, s2, s3] for s1 in [-1, 1]
                              for s2 in [-1, 1] for s3 in [-1, 1]],
                             dtype=np.float64)
            corners = c + (R @ (signs * (ext / 2)).T).T
            edges = [(0, 1), (1, 3), (3, 2), (2, 0),
                     (4, 5), (5, 7), (7, 6), (6, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            for a, b in edges:
                ax.plot([corners[a, 0], corners[b, 0]],
                        [corners[a, 1], corners[b, 1]],
                        [corners[a, 2], corners[b, 2]], 'g-', lw=1.5)
            ax.text(c[0], c[1], c[2], op["entity"], color='g', fontsize=8)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Scene reconstruction + camera frustums + object boxes')

    # equal aspect (matplotlib trick)
    if len(P):
        all_pts = P
        if obj_poses:
            obj_centers = np.array([op["center"] for op in obj_poses
                                    if op.get("status") == "ok"])
            if len(obj_centers):
                all_pts = np.vstack([all_pts, obj_centers])
        max_range = np.array([all_pts[:, 0].max() - all_pts[:, 0].min(),
                              all_pts[:, 1].max() - all_pts[:, 1].min(),
                              all_pts[:, 2].max() - all_pts[:, 2].min()]).max() / 2
        mid = all_pts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def save_pointcloud_ply(points, colors, path):
    """Save as PLY for inspection in MeshLab/Open3D/CloudCompare."""
    n = len(points)
    cols = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = cols[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")