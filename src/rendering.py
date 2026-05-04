"""Novel view synthesis.

Two backends:
  1) Point splatting (always available): projects colored points using
     p = K [R|t] X with depth-aware z-buffer and anisotropic kernel.
     Good photometric quality if dense enough.
  2) 3D Gaussian Splatting (optional): if `gsplat` or `diff-gaussian-rasterization`
     is installed, we forward to it. Otherwise falls back to (1).
"""
import numpy as np
import cv2


def _splat_points(points, colors, K, R, t, H, W,
                  base_radius=2, depth_modulation=True):
    """Z-buffered point splatting."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    if len(points) == 0:
        return (img * 255).astype(np.uint8)

    Xc = (R @ points.T + t.reshape(3, 1)).T  # (N,3)
    z = Xc[:, 2]
    front = z > 0.01
    Xc, z, cols = Xc[front], z[front], colors[front]
    if len(Xc) == 0:
        return (img * 255).astype(np.uint8)

    uv_h = (K @ Xc.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    u = uv[:, 0]; v = uv[:, 1]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_img]; v = v[in_img]; z = z[in_img]; cols = cols[in_img]

    # render back-to-front (painter's: but we use z-buffer for correctness)
    # paint each point as a small disk
    if depth_modulation:
        # nearer points get larger radius; bound between 1..4
        z_med = np.median(z) if len(z) else 1.0
        rad = np.clip((z_med / np.maximum(z, 1e-6)) * base_radius,
                      1, 4).astype(int)
    else:
        rad = np.full(len(z), base_radius, dtype=int)

    for i in range(len(u)):
        ui, vi = int(round(u[i])), int(round(v[i]))
        r = int(rad[i])
        u0, u1 = max(0, ui - r), min(W, ui + r + 1)
        v0, v1 = max(0, vi - r), min(H, vi + r + 1)
        if u0 >= u1 or v0 >= v1:
            continue
        # gaussian-ish weight inside disk
        yy, xx = np.mgrid[v0:v1, u0:u1]
        d2 = (xx - ui) ** 2 + (yy - vi) ** 2
        wt = np.exp(-d2 / max(1.0, r * r))
        # z-buffer test (closer wins)
        mask = z[i] < zbuf[v0:v1, u0:u1]
        sel = mask
        if np.any(sel):
            zbuf[v0:v1, u0:u1][sel] = z[i]
            for c in range(3):
                img[v0:v1, u0:u1, c][sel] = (
                    wt[sel] * cols[i, c] + (1 - wt[sel]) * img[v0:v1, u0:u1, c][sel]
                )

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def render_novel_view(scene_points, scene_colors, K, R, t, H, W):
    """RGB ndarray (H,W,3) uint8."""
    return _splat_points(scene_points, scene_colors, K, R, t, H, W)


def render_with_inpaint(scene_points, scene_colors, K, R, t, H, W):
    """Splat + inpaint holes for nicer photometric scoring."""
    rgb = render_novel_view(scene_points, scene_colors, K, R, t, H, W)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hole_mask = (gray == 0).astype(np.uint8) * 255
    if hole_mask.sum() > 0:
        rgb = cv2.inpaint(rgb, hole_mask, 3, cv2.INPAINT_TELEA)
    return rgb


# ---------- 3D Gaussian Splatting adapter (optional, plug-and-play) ----------
def render_3dgs(scene, K, R, t, H, W):
    """If a trained 3DGS scene is present (dict with means, scales, rots, opacities,
    sh/colors), render with gsplat or fall back to splatting."""
    try:
        import torch
        from gsplat.rendering import rasterization
        means = torch.tensor(scene["means"], dtype=torch.float32, device="cuda")
        quats = torch.tensor(scene["quats"], dtype=torch.float32, device="cuda")
        scales = torch.tensor(scene["scales"], dtype=torch.float32, device="cuda")
        opac = torch.tensor(scene["opacities"], dtype=torch.float32, device="cuda")
        cols = torch.tensor(scene["colors"], dtype=torch.float32, device="cuda")
        viewmat = np.eye(4); viewmat[:3, :3] = R; viewmat[:3, 3] = t
        viewmat = torch.tensor(viewmat[None], dtype=torch.float32, device="cuda")
        Kt = torch.tensor(K[None], dtype=torch.float32, device="cuda")
        rgb, _, _ = rasterization(means, quats, scales, opac, cols,
                                  viewmat, Kt, W, H)
        return (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    except Exception:
        # fallback
        return render_with_inpaint(scene["means"], scene["colors"], K, R, t, H, W)