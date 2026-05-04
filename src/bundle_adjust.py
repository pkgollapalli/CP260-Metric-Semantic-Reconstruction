"""Bundle adjustment: refine 3D points (poses optionally fixed) by
minimizing reprojection error across all views.

Implementation: scipy.optimize.least_squares with sparse Jacobian.
This realizes the MAP estimate via NLLS as derived in the factor graph
lecture (Lecture 12).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def ba_refine_points(frames, points, tracks, fix_poses=True, verbose=False):
    """Refine 3D point positions only (poses fixed).
    tracks: list of {frame_idx: (u,v)} aligned with `points`.
    Returns refined points.
    """
    if len(points) == 0:
        return points

    by_idx = {f["idx"]: f for f in frames}

    # Build observation list: (point_id, frame_idx, u, v)
    obs = []
    for pid, tr in enumerate(tracks):
        for fid, uv in tr.items():
            if fid in by_idx:
                obs.append((pid, fid, uv[0], uv[1]))
    if not obs:
        return points
    obs = np.array(obs, dtype=object)

    n_pts = len(points)
    x0 = points.reshape(-1).astype(np.float64)

    def residuals(x):
        pts = x.reshape(n_pts, 3)
        res = np.zeros(2 * len(obs))
        for k, (pid, fid, u, v) in enumerate(obs):
            f = by_idx[fid]
            X = pts[int(pid)]
            Xc = f["R"] @ X + f["t"]
            if Xc[2] <= 1e-6:
                res[2 * k] = 1e3; res[2 * k + 1] = 1e3
                continue
            uvp = f["K"] @ Xc
            up, vp = uvp[0] / uvp[2], uvp[1] / uvp[2]
            res[2 * k] = up - u
            res[2 * k + 1] = vp - v
        return res

    # Sparsity pattern: each obs depends only on its 3D point (3 params)
    m = 2 * len(obs)
    n = 3 * n_pts
    A = lil_matrix((m, n), dtype=int)
    for k, (pid, _, _, _) in enumerate(obs):
        pid = int(pid)
        for d in range(3):
            A[2 * k, 3 * pid + d] = 1
            A[2 * k + 1, 3 * pid + d] = 1

    res0 = residuals(x0)
    rms0 = float(np.sqrt(np.mean(res0 ** 2)))
    if verbose:
        print(f"[BA] initial reproj RMS = {rms0:.3f} px over {len(obs)} obs")

    sol = least_squares(residuals, x0, jac_sparsity=A,
                        method="trf", loss="huber", f_scale=2.0,
                        max_nfev=50, verbose=2 if verbose else 0)
    pts_refined = sol.x.reshape(n_pts, 3)
    res1 = residuals(sol.x)
    rms1 = float(np.sqrt(np.mean(res1 ** 2)))
    if verbose:
        print(f"[BA] final   reproj RMS = {rms1:.3f} px")
    return pts_refined