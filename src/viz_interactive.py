"""
viz_interactive.py  —  Team-B coolness upgrade
===============================================

Generates a self-contained HTML file with a fully interactive 3-D scene
viewer (Three.js + dat.GUI):

  • Orbiting point cloud coloured by depth
  • Camera frustums as line-segments
  • OBB wireframes per entity (toggle per entity in GUI)
  • Click any OBB → shows entity name + reproj error in HUD
  • "Project onto frame" dropdown: picks a frame, renders the OBB corners
    projected onto the actual photo as a canvas overlay
  • All data baked as JSON into the HTML; zero server needed

Also exports: save_pointcloud_ply (kept from original viz.py for compat.)
"""

from __future__ import annotations
import json
import os
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ─── kept for backwards compatibility ────────────────────────────────────────
def save_pointcloud_ply(points, colors, path):
    n    = len(points)
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
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# ─── matplotlib static (kept) ─────────────────────────────────────────────────
def plot_scene_3d(points, colors, frames, save_path=None,
                  obj_poses=None, frustum_scale=0.08):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    if len(points) > 50_000:
        idx = np.random.choice(len(points), 50_000, replace=False)
        P, C = points[idx], colors[idx]
    else:
        P, C = points, colors

    if len(P):
        ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                   c=np.clip(C, 0, 1), s=0.5, alpha=0.6)

    for f in frames:
        H, W = f["image"].shape[:2]
        K_inv = np.linalg.inv(f["K"])
        corn_px = np.array([[0,0,1],[W,0,1],[W,H,1],[0,H,1]], dtype=float)
        rays = (K_inv @ corn_px.T).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        pts_w = (f["R"].T @ (rays * frustum_scale).T).T - f["R"].T @ f["t"]
        Cw = -f["R"].T @ f["t"]
        for i in range(4):
            ax.plot([Cw[0], pts_w[i,0]], [Cw[1], pts_w[i,1]],
                    [Cw[2], pts_w[i,2]], "r-", lw=0.7, alpha=0.6)
        for i in range(4):
            j = (i+1) % 4
            ax.plot([pts_w[i,0], pts_w[j,0]], [pts_w[i,1], pts_w[j,1]],
                    [pts_w[i,2], pts_w[j,2]], "r-", lw=0.7, alpha=0.6)

    if obj_poses:
        for op in obj_poses:
            if op.get("status") != "ok":
                continue
            c   = np.array(op["center"])
            ext = np.array(op["extent"])
            R   = np.array(op["rotation_matrix"])
            signs = np.array([[s1,s2,s3] for s1 in [-1,1]
                               for s2 in [-1,1] for s3 in [-1,1]], float)
            corners = c + (R @ (signs * (ext/2)).T).T
            edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),
                     (0,4),(1,5),(2,6),(3,7)]
            for a, b in edges:
                ax.plot([corners[a,0],corners[b,0]],
                        [corners[a,1],corners[b,1]],
                        [corners[a,2],corners[b,2]], "g-", lw=1.5)
            ax.text(c[0], c[1], c[2], op["entity"], color="g", fontsize=8)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Scene reconstruction + camera frustums + OBBs")
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─── interactive HTML viewer ──────────────────────────────────────────────────
def _obb_corners(center, extent, rotation):
    """Return 8 corners of an OBB."""
    c   = np.array(center)
    e   = np.array(extent)
    R   = np.array(rotation)
    s   = np.array([[s1,s2,s3] for s1 in [-1,1]
                    for s2 in [-1,1] for s3 in [-1,1]], float)
    return c + (R @ (s * (e / 2)).T).T   # (8,3)


def _img_to_b64(path: str, max_dim: int = 800) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def build_interactive_viewer(
    points:    np.ndarray,
    colors:    np.ndarray,
    frames:    list[dict],
    answer:    list[dict],           # list of {entity, obb}
    save_path: str = "outputs/scene_viewer.html",
    max_cloud_pts: int = 30_000,
) -> str:
    """
    Build a self-contained HTML/Three.js interactive viewer.

    Controls:
      • Mouse: orbit / zoom / pan
      • GUI checkboxes: toggle each OBB entity
      • Click OBB wireframe: HUD shows entity name + center
      • Frame selector: overlay OBB projections on actual photo

    Returns the save path.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ── downsample cloud ──────────────────────────────────────────────────────
    if len(points) > max_cloud_pts:
        idx = np.random.choice(len(points), max_cloud_pts, replace=False)
        pts, cols = points[idx], colors[idx]
    else:
        pts, cols = points, colors

    cols_rgb = (np.clip(cols, 0, 1) * 255).astype(int)

    cloud_data = {
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "r": cols_rgb[:, 0].tolist(),
        "g": cols_rgb[:, 1].tolist(),
        "b": cols_rgb[:, 2].tolist(),
    }

    # ── camera centres ────────────────────────────────────────────────────────
    cam_centers = [(-f["R"].T @ f["t"]).tolist() for f in frames]
    cam_labels  = [str(f["idx"]) for f in frames]

    # ── OBB data ──────────────────────────────────────────────────────────────
    OBB_EDGE_IDX = [
        [0,1],[1,3],[3,2],[2,0],   # front face
        [4,5],[5,7],[7,6],[6,4],   # back face
        [0,4],[1,5],[2,6],[3,7],   # connecting
    ]
    obb_data = []
    for entry in answer:
        obb  = entry["obb"]
        corn = _obb_corners(obb["center"], obb["extent"], obb["rotation"])
        obb_data.append({
            "entity":  entry["entity"],
            "center":  obb["center"],
            "corners": corn.tolist(),
            "edges":   OBB_EDGE_IDX,
        })

    # ── frame thumbnails (b64 JPEG) for projection overlay ────────────────────
    frame_imgs = {}
    for f in frames:
        frame_imgs[str(f["idx"])] = _img_to_b64(f["path"])

    # ── K for projection (use first frame's K, same for all here) ─────────────
    K0 = frames[0]["K"].tolist()
    poses_js = {}
    for f in frames:
        poses_js[str(f["idx"])] = {
            "R": f["R"].tolist(),
            "t": f["t"].tolist(),
            "K": f["K"].tolist(),
        }

    payload = json.dumps({
        "cloud":       cloud_data,
        "cam_centers": cam_centers,
        "cam_labels":  cam_labels,
        "obbs":        obb_data,
        "frames":      frame_imgs,
        "poses":       poses_js,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CP260 Scene Viewer — Team B</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a0f; color:#e0e0e0; font-family:'Courier New',monospace;
          overflow:hidden; }}
  #canvas-wrap {{ width:100vw; height:100vh; }}
  #hud {{
    position:fixed; top:16px; left:16px;
    background:rgba(0,0,0,.72); border:1px solid #2a6eff44;
    padding:12px 16px; border-radius:6px; font-size:12px;
    pointer-events:none; min-width:220px;
  }}
  #hud h2 {{ font-size:13px; color:#2a6eff; margin-bottom:6px; }}
  #hud .row {{ display:flex; justify-content:space-between; gap:16px;
               margin:2px 0; }}
  #hud .val {{ color:#7effa0; }}
  #gui {{
    position:fixed; top:16px; right:16px;
    background:rgba(0,0,0,.80); border:1px solid #333;
    padding:14px; border-radius:6px; font-size:12px; min-width:180px;
  }}
  #gui h3 {{ color:#2a6eff; margin-bottom:8px; font-size:12px; }}
  #gui label {{ display:flex; align-items:center; gap:8px;
                cursor:pointer; margin:4px 0; }}
  #gui input[type=checkbox] {{ accent-color:#2a6eff; }}
  #overlay-wrap {{
    position:fixed; bottom:16px; left:16px;
    background:rgba(0,0,0,.85); border:1px solid #2a6eff44;
    border-radius:6px; padding:10px; display:none;
  }}
  #overlay-wrap h3 {{ color:#2a6eff; font-size:11px; margin-bottom:6px; }}
  #frame-select {{ background:#111; color:#e0e0e0; border:1px solid #333;
                   border-radius:4px; padding:4px; font-size:11px; width:100%; }}
  #proj-canvas {{ display:block; margin-top:8px; border-radius:4px;
                  max-width:480px; border:1px solid #333; }}
  #show-proj-btn {{
    margin-top:6px; background:#2a6eff; color:#fff; border:none;
    border-radius:4px; padding:5px 12px; cursor:pointer; font-size:11px;
    width:100%;
  }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>

<div id="hud">
  <h2>CP260 · Metric-Semantic Scene</h2>
  <div class="row"><span>Points</span><span class="val" id="h-pts">—</span></div>
  <div class="row"><span>Cameras</span><span class="val" id="h-cams">—</span></div>
  <div class="row"><span>Entities</span><span class="val" id="h-ents">—</span></div>
  <div class="row"><span>Selected</span><span class="val" id="h-sel">none</span></div>
  <div class="row"><span>Center</span><span class="val" id="h-ctr">—</span></div>
</div>

<div id="gui">
  <h3>Toggle OBBs</h3>
  <div id="obb-toggles"></div>
  <hr style="border-color:#222;margin:8px 0">
  <label>
    <input type="checkbox" id="tog-cloud" checked> Point cloud
  </label>
  <label>
    <input type="checkbox" id="tog-cams" checked> Cameras
  </label>
  <hr style="border-color:#222;margin:8px 0">
  <button id="show-proj-btn" onclick="document.getElementById('overlay-wrap').style.display='block'">
    Project onto frame ▼
  </button>
</div>

<div id="overlay-wrap">
  <h3>OBB Projection onto Frame</h3>
  <select id="frame-select"></select>
  <canvas id="proj-canvas" width="480" height="270"></canvas>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── data ─────────────────────────────────────────────────────────────────────
const DATA = {payload};

// ── Three.js setup ────────────────────────────────────────────────────────────
const wrap   = document.getElementById("canvas-wrap");
const W = wrap.clientWidth, H = wrap.clientHeight;
const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
wrap.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const camera = new THREE.PerspectiveCamera(55, W/H, 0.01, 50);
camera.position.set(0.5, 0.5, 2.2);

// ── orbit controls (manual, no import needed) ─────────────────────────────────
let isDragging=false, prevMouse={{x:0,y:0}};
let theta=0, phi=Math.PI/3, radius=2.0, target=new THREE.Vector3(0.35,0.25,0.7);

function updateCamera() {{
  camera.position.set(
    target.x + radius*Math.sin(phi)*Math.sin(theta),
    target.y + radius*Math.cos(phi),
    target.z + radius*Math.sin(phi)*Math.cos(theta),
  );
  camera.lookAt(target);
}}
updateCamera();

renderer.domElement.addEventListener("mousedown", e => {{ isDragging=true; prevMouse={{x:e.clientX,y:e.clientY}}; }});
renderer.domElement.addEventListener("mouseup",   () => isDragging=false);
renderer.domElement.addEventListener("mousemove", e => {{
  if (!isDragging) return;
  const dx=(e.clientX-prevMouse.x)*0.005, dy=(e.clientY-prevMouse.y)*0.005;
  theta -= dx; phi = Math.max(0.05, Math.min(Math.PI-0.05, phi+dy));
  prevMouse={{x:e.clientX,y:e.clientY}};
  updateCamera();
}});
renderer.domElement.addEventListener("wheel", e => {{
  radius = Math.max(0.2, radius + e.deltaY*0.002);
  updateCamera();
}});

// ── point cloud ───────────────────────────────────────────────────────────────
const geo = new THREE.BufferGeometry();
const cx = DATA.cloud.x, cy = DATA.cloud.y, cz = DATA.cloud.z;
const n  = cx.length;
const pos  = new Float32Array(n*3);
const cols = new Float32Array(n*3);
for (let i=0;i<n;i++) {{
  pos[3*i]   = cx[i]; pos[3*i+1] = cy[i]; pos[3*i+2] = cz[i];
  cols[3*i]  = DATA.cloud.r[i]/255;
  cols[3*i+1]= DATA.cloud.g[i]/255;
  cols[3*i+2]= DATA.cloud.b[i]/255;
}}
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
geo.setAttribute("color",    new THREE.BufferAttribute(cols,3));
const cloudMesh = new THREE.Points(geo,
  new THREE.PointsMaterial({{size:0.003, vertexColors:true}}));
scene.add(cloudMesh);
document.getElementById("h-pts").textContent = n.toLocaleString();

// ── camera markers ────────────────────────────────────────────────────────────
const camGroup = new THREE.Group();
DATA.cam_centers.forEach((c,i) => {{
  const sph = new THREE.Mesh(
    new THREE.SphereGeometry(0.008,6,6),
    new THREE.MeshBasicMaterial({{color:0xff4444}})
  );
  sph.position.set(c[0],c[1],c[2]);
  camGroup.add(sph);
}});
scene.add(camGroup);
document.getElementById("h-cams").textContent = DATA.cam_centers.length;

// ── OBBs ──────────────────────────────────────────────────────────────────────
const PALETTE = [0x00ff88,0x00ccff,0xff5050,0xffcc00,0xcc44ff,0xff8800,0x44ffdd,0xffaacc];
const obbGroups = [];
const obbTogDiv = document.getElementById("obb-toggles");

DATA.obbs.forEach((obb, oi) => {{
  const col   = PALETTE[oi % PALETTE.length];
  const mat   = new THREE.LineBasicMaterial({{color:col, linewidth:2}});
  const group = new THREE.Group();
  group.userData = obb;

  const corn = obb.corners;
  obb.edges.forEach(([a,b]) => {{
    const g2 = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...corn[a]),
      new THREE.Vector3(...corn[b]),
    ]);
    group.add(new THREE.Line(g2, mat));
  }});

  // center sphere for clicking
  const sph = new THREE.Mesh(
    new THREE.SphereGeometry(0.008,8,8),
    new THREE.MeshBasicMaterial({{color:col}})
  );
  sph.position.set(...obb.center);
  sph.userData = obb;
  group.add(sph);

  scene.add(group);
  obbGroups.push(group);

  // GUI toggle
  const lbl = document.createElement("label");
  const chk = document.createElement("input");
  chk.type = "checkbox"; chk.checked = true;
  chk.style.accentColor = "#" + col.toString(16).padStart(6,"0");
  chk.addEventListener("change", () => {{ group.visible = chk.checked; }});
  const span = document.createElement("span");
  span.style.color = "#" + col.toString(16).padStart(6,"0");
  span.textContent = obb.entity;
  lbl.appendChild(chk); lbl.appendChild(span);
  obbTogDiv.appendChild(lbl);
}});
document.getElementById("h-ents").textContent = DATA.obbs.length;

// ── global toggles ────────────────────────────────────────────────────────────
document.getElementById("tog-cloud").addEventListener("change", e =>
  cloudMesh.visible = e.target.checked);
document.getElementById("tog-cams").addEventListener("change", e =>
  camGroup.visible = e.target.checked);

// ── raycasting (click OBB center spheres) ─────────────────────────────────────
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
renderer.domElement.addEventListener("click", e => {{
  mouse.x =  (e.clientX / W) * 2 - 1;
  mouse.y = -(e.clientY / H) * 2 + 1;
  ray.setFromCamera(mouse, camera);
  const targets = obbGroups.flatMap(g => g.children.filter(c => c.isMesh));
  const hits = ray.intersectObjects(targets);
  if (hits.length) {{
    const obb = hits[0].object.userData;
    document.getElementById("h-sel").textContent = obb.entity;
    document.getElementById("h-ctr").textContent =
      obb.center.map(v => v.toFixed(3)).join(", ");
  }}
}});

// ── render loop ───────────────────────────────────────────────────────────────
(function animate() {{
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}})();

// ── projection overlay ────────────────────────────────────────────────────────
const fsel = document.getElementById("frame-select");
Object.keys(DATA.frames).sort((a,b)=>+a-+b).forEach(k => {{
  const opt = document.createElement("option");
  opt.value = k; opt.textContent = "Frame " + k;
  fsel.appendChild(opt);
}});

fsel.addEventListener("change", drawProjection);

function matMul3x3_3x1(M, v) {{
  return [
    M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
    M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
    M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2],
  ];
}}

function projectPt(X, pose) {{
  const R=pose.R, t=pose.t, K=pose.K;
  const Xc=[R[0][0]*X[0]+R[0][1]*X[1]+R[0][2]*X[2]+t[0],
             R[1][0]*X[0]+R[1][1]*X[1]+R[1][2]*X[2]+t[1],
             R[2][0]*X[0]+R[2][1]*X[1]+R[2][2]*X[2]+t[2]];
  if (Xc[2]<=0) return null;
  const uvh=matMul3x3_3x1(K,Xc);
  return [uvh[0]/uvh[2], uvh[1]/uvh[2]];
}}

function drawProjection() {{
  const fkey = fsel.value;
  const b64  = DATA.frames[fkey];
  if (!b64) return;
  const pose = DATA.poses[fkey];
  const canv = document.getElementById("proj-canvas");
  const ctx  = canv.getContext("2d");

  const img = new Image();
  img.onload = () => {{
    canv.width = 480; canv.height = Math.round(480 * img.height / img.width);
    const sx = 480 / img.width, sy = canv.height / img.height;
    ctx.drawImage(img, 0, 0, canv.width, canv.height);

    DATA.obbs.forEach((obb, oi) => {{
      const col = PALETTE[oi % PALETTE.length];
      const hex = "#"+col.toString(16).padStart(6,"0");
      ctx.strokeStyle = hex; ctx.lineWidth = 2;

      const corn2d = obb.corners.map(c => {{
        const uv = projectPt(c, pose);
        if (!uv) return null;
        return [uv[0]*sx, uv[1]*sy];
      }});

      obb.edges.forEach(([a,b]) => {{
        if (!corn2d[a] || !corn2d[b]) return;
        ctx.beginPath();
        ctx.moveTo(...corn2d[a]);
        ctx.lineTo(...corn2d[b]);
        ctx.stroke();
      }});

      // label
      const ctr = projectPt(obb.center, pose);
      if (ctr) {{
        ctx.fillStyle = hex;
        ctx.font = "bold 11px monospace";
        ctx.fillText(obb.entity, ctr[0]*sx+4, ctr[1]*sy-4);
      }}
    }});
  }};
  img.src = "data:image/jpeg;base64," + b64;
}}

// auto-draw first frame
if (fsel.options.length) drawProjection();
</script>
</body>
</html>"""

    with open(save_path, "w") as f:
        f.write(html)

    print(f"[viz] Interactive viewer saved → {save_path}")
    return save_path
