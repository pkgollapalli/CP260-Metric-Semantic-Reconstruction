"""6-DoF object pose estimation.

For each target object (e.g. "power socket"):
  1. Detect across all views using GroundingDINO.
  2. Match detections across views (geometric: bbox centers epipolar-consistent
     OR simply pick highest-confidence pair as anchor and triangulate centers).
  3. Triangulate the bbox center across all views -> object center in world.
  4. For orientation:
     - take all 3D scene points whose 2D projection falls inside the object
       bbox in each view (intersection across views).
     - run PCA on those points: principal axes give orientation,
       eigenvalues give extent.
  5. Output: center (3,), extent (3,), R (3,3) / quaternion (4,).
"""
import numpy as np
import cv2
from .geometry import triangulate_multi_view, project_points
from .data_loader import R_to_qvec


def collect_observations(frames, detections, label_substr):
    """Pick best detection per frame matching label_substr. Returns
    list of (frame_idx, bbox, center_uv, score)."""
    obs = []
    for f in frames:
        dets = detections.get(f["idx"], [])
        cands = [d for d in dets if label_substr in d["label"]]
        if not cands:
            continue
        d = max(cands, key=lambda x: x["score"])
        x0, y0, x1, y1 = d["bbox"]
        obs.append({
            "idx": f["idx"],
            "bbox": d["bbox"],
            "center": (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
            "score": d["score"],
        })
    return obs


def triangulate_center(frames, observations, ransac_iters=50, inlier_px=8.0):
    """Robust multi-view triangulation of object center.
    RANSAC over pairs to handle outlier detections."""
    if len(observations) < 2:
        return None, []

    by_idx = {f["idx"]: f for f in frames}
    obs = observations

    rng = np.random.default_rng(0)
    best = (np.inf, None, [])

    n = len(obs)
    if n == 2:
        Ps = [by_idx[obs[0]["idx"]]["P"], by_idx[obs[1]["idx"]]["P"]]
        uvs = [obs[0]["center"], obs[1]["center"]]
        X, rms = triangulate_multi_view(Ps, uvs)
        return X, [0, 1] if rms < np.inf else []

    for _ in range(ransac_iters):
        i, j = rng.choice(n, size=2, replace=False)
        Ps = [by_idx[obs[i]["idx"]]["P"], by_idx[obs[j]["idx"]]["P"]]
        uvs = [obs[i]["center"], obs[j]["center"]]
        X, _ = triangulate_multi_view(Ps, uvs)
        # check consistency on all views
        inliers = []
        for k, o in enumerate(obs):
            f = by_idx[o["idx"]]
            uv_p, z = project_points(X.reshape(1, 3), f["K"], f["R"], f["t"])
            if z[0] <= 0:
                continue
            err = np.linalg.norm(uv_p[0] - np.array(o["center"]))
            if err < inlier_px:
                inliers.append(k)
        if len(inliers) >= 2:
            # refit on inliers
            Ps = [by_idx[obs[k]["idx"]]["P"] for k in inliers]
            uvs = [obs[k]["center"] for k in inliers]
            X, rms = triangulate_multi_view(Ps, uvs)
            if rms < best[0]:
                best = (rms, X, inliers)

    if best[1] is None:
        # fallback: top-2 by score
        top = sorted(range(n), key=lambda k: -obs[k]["score"])[:2]
        Ps = [by_idx[obs[k]["idx"]]["P"] for k in top]
        uvs = [obs[k]["center"] for k in top]
        X, _ = triangulate_multi_view(Ps, uvs)
        return X, top

    return best[1], best[2]


def points_in_object(scene_points, frames, observations, inlier_views=2):
    """Find scene 3D points whose 2D projection falls inside the bbox in at
    least `inlier_views` of the views with that detection."""
    if len(scene_points) == 0 or len(observations) == 0:
        return np.zeros((0, 3))
    by_idx = {f["idx"]: f for f in frames}
    inside_count = np.zeros(len(scene_points), dtype=int)
    for o in observations:
        f = by_idx[o["idx"]]
        uv, z = project_points(scene_points, f["K"], f["R"], f["t"])
        x0, y0, x1, y1 = o["bbox"]
        ok = (z > 0) & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) \
             & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
        inside_count += ok.astype(int)
    keep = inside_count >= min(inlier_views, len(observations))
    return scene_points[keep]


def fit_oriented_box(points_3d, fallback_extent=0.05):
    """PCA -> oriented box. Returns center (3,), extent (3,), R (3,3)."""
    if len(points_3d) < 3:
        return None
    c = points_3d.mean(axis=0)
    Q = points_3d - c
    cov = Q.T @ Q / max(1, len(Q) - 1)
    w, V = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; sort descending
    order = np.argsort(-w)
    w = w[order]; V = V[:, order]
    # ensure right-handed
    if np.linalg.det(V) < 0:
        V[:, -1] = -V[:, -1]
    # extent = 2 * std along each axis (≈ box half-size doubled). Use 4*sigma
    # so that ~95% of points are inside.
    extent = np.sqrt(np.maximum(w, 1e-8)) * 4.0
    extent = np.maximum(extent, fallback_extent)
    return c, extent, V


def estimate_object_pose(label_name, prompts, frames, scene_points, scene_colors,
                        detections):
    """End-to-end object pose. Returns dict ready to dump as JSON."""
    # which substring to look for in detected labels
    label_key = prompts[0].split()[0].lower()  # e.g. "power" or "ethernet"
    obs = collect_observations(frames, detections, label_key)
    if len(obs) < 2:
        # try second token
        if len(prompts[0].split()) > 1:
            label_key2 = prompts[0].split()[1].lower()
            obs = collect_observations(frames, detections, label_key2)
    if len(obs) < 2:
        return {
            "entity": label_name,
            "status": "insufficient_detections",
            "n_views": len(obs),
        }

    center, inliers = triangulate_center(frames, obs)
    if center is None:
        return {"entity": label_name, "status": "triangulation_failed",
                "n_views": len(obs)}

    inlier_obs = [obs[k] for k in inliers] if inliers else obs

    # orientation/extent from scene points inside the object bbox
    pts_in = points_in_object(scene_points, frames, inlier_obs,
                              inlier_views=max(2, len(inlier_obs) // 2))
    if len(pts_in) >= 3:
        c2, extent, R_obj = fit_oriented_box(pts_in)
        # use triangulated center (more stable for 1-2 cluster of points)
        center_final = center if np.linalg.norm(center - c2) < np.linalg.norm(extent) else c2
    else:
        # not enough scene points -> isotropic small extent, identity R
        extent = np.array([0.05, 0.05, 0.02])
        R_obj = np.eye(3)
        center_final = center

    quat = R_to_qvec(R_obj)  # (w,x,y,z)

    return {
        "entity": label_name,
        "center": [float(x) for x in center_final],
        "extent": [float(x) for x in extent],
        "rotation_matrix": R_obj.tolist(),
        "quaternion_wxyz": [float(x) for x in quat],
        "n_views": len(obs),
        "n_inlier_views": len(inlier_obs),
        "n_scene_points_inside": int(len(pts_in)),
        "status": "ok",
    }