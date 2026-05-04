"""Geometry primitives.

p = K [R | t] X    (world -> camera -> image)

Triangulation: linear DLT then nonlinear refine.
"""
import numpy as np
import cv2


def project_points(X_world, K, R, t):
    """Project 3D world points to 2D pixels.
    X_world: (N,3); returns (N,2), depth (N,)."""
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    Xc = (R @ X.T + t.reshape(3, 1)).T  # (N,3)
    z = Xc[:, 2]
    uv_h = (K @ Xc.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    return uv, z


def triangulate_two_view(P1, P2, pts1, pts2):
    """Linear DLT triangulation; returns (N,3) world points."""
    pts1 = np.asarray(pts1, dtype=np.float64).T  # (2,N)
    pts2 = np.asarray(pts2, dtype=np.float64).T
    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)  # (4,N)
    X = (X_h[:3] / X_h[3:]).T
    return X


def triangulate_multi_view(Ps, uvs):
    """N-view DLT triangulation of one 3D point.
    Ps: list of (3,4) projection matrices.
    uvs: list of (u,v) for that point in each view.
    Returns: X (3,) and reproj rms.
    """
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]
    # reproj rms
    err = []
    for P, (u, v) in zip(Ps, uvs):
        x = P @ np.append(X, 1.0)
        x = x[:2] / x[2]
        err.append(np.linalg.norm(x - np.array([u, v])))
    rms = float(np.sqrt(np.mean(np.square(err))))
    return X, rms


def reprojection_error(X, K, R, t, uv_obs):
    """Single-point reprojection error (px)."""
    uv_pred, _ = project_points(X.reshape(1, 3), K, R, t)
    return float(np.linalg.norm(uv_pred[0] - uv_obs))


def filter_by_chirality(X, R, t):
    """Keep points with positive depth in the camera (in front of camera)."""
    Xc = R @ X.T + t.reshape(3, 1)
    return Xc[2] > 0