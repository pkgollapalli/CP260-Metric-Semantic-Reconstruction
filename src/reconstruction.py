"""Sparse and dense reconstruction.

Sparse: ORB + cross-check matching + Lowe ratio + multi-view triangulation.
Dense: depth estimation via plane-sweep / per-pixel multi-view stereo on
       a small grid (used to seed 3DGS or for direct rendering).
"""
import numpy as np
import cv2
from collections import defaultdict
from .geometry import triangulate_multi_view


# ---------- ORB feature extraction & matching ----------
class FeatureBank:
    def __init__(self, n_features=4000):
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7,
                                  scaleFactor=1.2, nlevels=8)
        self.keypoints = {}     # idx -> [cv2.KeyPoint]
        self.descriptors = {}   # idx -> ndarray (N, 32) uint8

    def extract(self, frames):
        for f in frames:
            gray = cv2.cvtColor(f["image"], cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.keypoints[f["idx"]] = kp
            self.descriptors[f["idx"]] = des
        return self


def match_pair(desA, desB, ratio=0.75):
    """BFMatcher with Lowe ratio test for ORB (Hamming)."""
    if desA is None or desB is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desA, desB, k=2)
    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good


def build_tracks(frames, fb, ratio=0.75, geometric_check=True):
    """Build feature tracks across multiple views via union-find on matches.
    Returns: tracks = list of {frame_idx -> (u,v)} dicts (>= 2 views per track).
    """
    # union-find over (frame_idx, kp_idx)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # all keypoint nodes
    for f in frames:
        for ki in range(len(fb.keypoints[f["idx"]])):
            parent[(f["idx"], ki)] = (f["idx"], ki)

    # pairwise matching between every pair (small N here -> ok)
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = frames[i], frames[j]
            matches = match_pair(fb.descriptors[fi["idx"]],
                                 fb.descriptors[fj["idx"]], ratio)
            if len(matches) < 8:
                continue

            if geometric_check:
                kpi = fb.keypoints[fi["idx"]]
                kpj = fb.keypoints[fj["idx"]]
                pi = np.float32([kpi[a].pt for a, _ in matches])
                pj = np.float32([kpj[b].pt for _, b in matches])
                # Fundamental matrix RANSAC to drop outliers
                F, mask = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, 2.0, 0.99)
                if mask is None:
                    continue
                matches = [m for m, ok in zip(matches, mask.ravel()) if ok]

            for a, b in matches:
                union((fi["idx"], a), (fj["idx"], b))

    # group by root
    groups = defaultdict(list)
    for node, _ in parent.items():
        groups[find(node)].append(node)

    tracks = []
    for nodes in groups.values():
        if len(nodes) < 2:
            continue
        # at most one obs per frame
        per_frame = {}
        for (fid, ki) in nodes:
            kp = fb.keypoints[fid]
            if fid in per_frame:
                continue
            per_frame[fid] = (kp[ki].pt[0], kp[ki].pt[1])
        if len(per_frame) >= 2:
            tracks.append(per_frame)
    return tracks


def triangulate_tracks(frames, tracks, max_reproj_err=4.0):
    """Triangulate every track using all observing cameras.
    Returns: X (M,3), per_track_views (list of dicts), per_track_color."""
    by_idx = {f["idx"]: f for f in frames}
    pts3d, colors, kept_tracks = [], [], []
    for tr in tracks:
        Ps, uvs, cols = [], [], []
        for fid, uv in tr.items():
            f = by_idx[fid]
            Ps.append(f["P"])
            uvs.append(uv)
            u, v = int(round(uv[0])), int(round(uv[1]))
            H, W = f["image"].shape[:2]
            if 0 <= u < W and 0 <= v < H:
                bgr = f["image"][v, u]
                cols.append(bgr[::-1])  # BGR->RGB
        X, rms = triangulate_multi_view(Ps, uvs)
        if rms > max_reproj_err:
            continue
        # cheirality: must be in front of every observing camera
        ok = True
        for fid in tr.keys():
            f = by_idx[fid]
            Xc = f["R"] @ X + f["t"]
            if Xc[2] <= 0:
                ok = False
                break
        if not ok:
            continue
        pts3d.append(X)
        colors.append(np.mean(cols, axis=0) if cols else np.array([200, 200, 200]))
        kept_tracks.append(tr)
    if not pts3d:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    return np.array(pts3d), np.array(colors) / 255.0, kept_tracks


# ---------- dense densification (cheap MVS-lite) ----------
def densify_with_grid(frames, sparse_pts, target_pts=200_000, grid_step=4):
    """Augment sparse cloud by lifting pixels to 3D using ALL views' depths
    via plane-sweep around sparse points' depth range. Cheap fallback.

    Strategy: for each ref frame, sample pixels on a grid; for each pixel
    pick the depth (from a discrete set spanning sparse depth percentile range)
    that minimizes SSD across neighbor views. Keep best matches.
    """
    if len(sparse_pts) < 8:
        return sparse_pts, np.full((len(sparse_pts), 3), 0.7)

    # depth range from sparse cloud projected to first frame
    f0 = frames[0]
    Xc = (f0["R"] @ sparse_pts.T + f0["t"].reshape(3, 1)).T
    z = Xc[:, 2]
    z_min, z_max = np.percentile(z, 5), np.percentile(z, 95)
    if z_min <= 0:
        z_min = 0.05
    depths = np.linspace(z_min, z_max, 32)

    H, W = f0["image"].shape[:2]
    K_ref = f0["K"]; K_inv = np.linalg.inv(K_ref)

    new_pts, new_cols = [], []
    half = 2  # 5x5 patch
    # neighbor views for photoconsistency
    ref = f0
    nbrs = frames[1:min(5, len(frames))]

    grays = {f["idx"]: cv2.cvtColor(f["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
             for f in [ref] + nbrs}
    g_ref = grays[ref["idx"]]

    R_ref, t_ref = ref["R"], ref["t"]
    for v in range(half, H - half, grid_step):
        for u in range(half, W - half, grid_step):
            patch_ref = g_ref[v - half:v + half + 1, u - half:u + half + 1]
            best_err, best_d = np.inf, None
            ray_cam = K_inv @ np.array([u, v, 1.0])
            for d in depths:
                Xc_pt = ray_cam * d
                Xw = R_ref.T @ (Xc_pt - t_ref)
                err_acc, count = 0.0, 0
                for nb in nbrs:
                    Xn = nb["R"] @ Xw + nb["t"]
                    if Xn[2] <= 0:
                        continue
                    uv_h = nb["K"] @ Xn
                    un, vn = uv_h[0] / uv_h[2], uv_h[1] / uv_h[2]
                    if not (half <= un < W - half and half <= vn < H - half):
                        continue
                    un_i, vn_i = int(round(un)), int(round(vn))
                    patch_nb = grays[nb["idx"]][vn_i - half:vn_i + half + 1,
                                                un_i - half:un_i + half + 1]
                    if patch_nb.shape != patch_ref.shape:
                        continue
                    err_acc += float(np.mean((patch_ref - patch_nb) ** 2))
                    count += 1
                if count >= 1:
                    err_acc /= count
                    if err_acc < best_err:
                        best_err, best_d = err_acc, d
            if best_d is None or best_err > 600:  # threshold to keep good matches
                continue
            ray_cam = K_inv @ np.array([u, v, 1.0])
            Xc_pt = ray_cam * best_d
            Xw = R_ref.T @ (Xc_pt - t_ref)
            new_pts.append(Xw)
            new_cols.append(ref["image"][v, u][::-1] / 255.0)
            if len(new_pts) >= target_pts:
                break
        if len(new_pts) >= target_pts:
            break

    if not new_pts:
        return sparse_pts, np.full((len(sparse_pts), 3), 0.7)
    new_pts = np.array(new_pts)
    new_cols = np.array(new_cols)
    return new_pts, new_cols


def reconstruct(frames, n_features=4000, ratio=0.75, max_reproj_err=4.0):
    """Full sparse reconstruction pipeline."""
    fb = FeatureBank(n_features=n_features).extract(frames)
    print(f"[recon] features extracted in {len(frames)} frames")
    tracks = build_tracks(frames, fb, ratio=ratio)
    print(f"[recon] {len(tracks)} multi-view tracks")
    X, colors, kept_tracks = triangulate_tracks(frames, tracks,
                                                max_reproj_err=max_reproj_err)
    print(f"[recon] {len(X)} 3D points after cheirality + reproj filter")
    return {
        "feature_bank": fb,
        "tracks": kept_tracks,
        "points": X,
        "colors": colors,
    }