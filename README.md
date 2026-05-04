# CP260-2026 Final Project — Metric-Semantic 3D Reconstruction

**Author:** Gollapalli Praveen Kumar (SR No. 27480) 

**Course:** CP260 Robotic Perception, IISc, 2026 

**Course Instructor:** Professor. Bharadwaj Amrutur

**Submission file:** `outputs/answer_final.json`

---

## Problem

Sixteen posed RGB images of a desktop PC tower are given. The task is to produce 6-DoF
Oriented Bounding Boxes (OBBs) for the VGA, Ethernet, and Power sockets on the rear I/O
panel. Grading is by 2D polygonal IoU between the projected OBB and a ground-truth
bounding box on a held-out test set. The pipeline must also generalise to arbitrary
entity prompts on evaluation day without retraining.

---

## Results

`outputs/answer_final.json` — three baseline entities:

| Entity            | Centre (m)                          | Extent (m)                    | Max reproj. error |
|-------------------|--------------------------------------|-------------------------------|-------------------|
| `vga_socket`      | (0.2705, 0.2261, 0.8349) — GT       | (0.0354, 0.0118, 0.0061) — GT | 0.0 px            |
| `ethernet_socket` | (0.2855, 0.2311, 0.7549)            | (0.0160, 0.0135, 0.0138) IEC  | 1.4 px            |
| `power_socket`    | (0.2921, 0.2190, 0.5291)            | (0.0278, 0.0198, 0.0280) IEC  | 2.2 px            |

`outputs/answer_eval_day.json` — five entities for evaluation day (3 baseline + 2 bonus):

| Entity            | Notes                                           |
|-------------------|-------------------------------------------------|
| `vga_socket`      | GT reference (identical to answer_final.json)   |
| `ethernet_socket` | IEC 60603-7 RJ45 extent, reproj ≤ 1.4 px       |
| `power_socket`    | IEC 60320 C14 extent, reproj ≤ 2.2 px          |
| `dvi_port`        | Bonus — DVI-I dual-link, 39 × 14 mm footprint  |
| `usb3_port`       | Bonus — USB-A 3.0, 14 × 6.5 mm footprint       |

All five share the GT VGA rotation matrix (same physical panel plane).  
Projected visualisations for the three baseline entities are in `docs/obb_debug/`.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── main.py                      ← full pipeline runner
├── eval_day_cookbook.py         ← evaluation-day script (PATH A + PATH B)
├── src/
│   ├── data_loader.py           ← image + pose loading; c2w ↔ w2c
│   ├── reconstruction.py        ← ORB SfM + multi-view DLT
│   ├── bundle_adjust.py         ← scipy BA (points only, poses fixed)
│   ├── detection.py             ← GroundingDINO + SAM mask lifting (★ upgraded)
│   ├── geometry.py              ← projection, DLT triangulation helpers
│   ├── pose_estimation.py       ← bbox-tri + PCA OBB fitter
│   ├── solve_obb.py             ← generalisable OBB solver
│   ├── build_answer.py          ← deterministic builder + JSON schema validator
│   ├── rendering.py             ← point splatting + Telea inpainting
│   ├── viz.py                   ← matplotlib 3D scene + static OBB plot
│   └── viz_interactive.py       ← Three.js self-contained HTML viewer (★ new)
├── outputs/
│   ├── answer_final.json        ← FINAL SUBMISSION (3 entities)
│   └── answer_eval_day.json     ← Eval-day file (5 entities: 3 baseline + 2 bonus)
└── docs/
    ├── report.pdf
    ├── report.md
    └── obb_debug/               ← OBB projected onto frames for VGA/ETH/PWR
        ├── frame_365.jpg
        ├── frame_449.jpg
        ├── frame_461.jpg
        └── ...
```

---

## Reproducing the submission JSON

**No-GPU path:**

```bash
pip install -r requirements.txt
python -m src.build_answer \
    --poses ./data/Data/poses.json \
    --K     1477.01,1480.44,1298.25,686.82 \
    --output outputs/answer_final.json
```

Runs in under 5 seconds on any CPU. No DINO, no point cloud required.

**Full GPU pipeline (ORB SfM + BA + DINO + OBB solver):**

```bash
python main.py \
    --data_dir ./data/Data \
    --poses    ./data/Data/poses.json \
    --K        1477.01,1480.44,1298.25,686.82 \
    --c2w \
    --output_dir outputs
```

Use `--no_dino` to skip DINO on CPU-only machines.

---

## Method

Each entity centre is recovered by **multi-view DLT triangulation** of manually verified pixel
correspondences across three frames. The three pairwise estimates are reduced via
coordinate-wise median, then **projected onto the I/O panel plane** defined by the GT VGA
rotation — eliminating depth error along the camera's line of sight (the dominant failure
mode in oblique views). Extents use IEC connector standard physical dimensions. The rotation
is shared with the GT VGA reference, since all rear-panel ports lie on the same physical plane.

For evaluation-day entities, an open-vocabulary **GroundingDINO** detector is used first
(PATH A); if detections are inconsistent, the same correspondence-based path is used with
two or three manually read pixels (PATH B). Both paths run through the same code in
`src/solve_obb.py` and `src/build_answer.py`.

---

## Detection upgrades (detection.py)

`src/detection.py` extends the baseline GroundingDINO detector with two improvements:

- **SAM ViT-H mask extraction.** For every DINO bounding box, a SAM prompt produces a
  pixel-accurate segmentation mask. This removes background points from the 3D lifting
  step, significantly tightening candidate point sets for small connectors.

- **Mask-guided 3D point lifting (`lift_mask_to_3d`).** Instead of casting a wide ray-cone
  from the bbox centre, each sparse-cloud point is projected into the frame and kept only
  if it falls inside the SAM mask. This yields a much tighter and more accurate set of 3D
  candidate points than cone-based lifting.

Both features are optional and fall back gracefully if SAM is not installed.

---

## Interactive 3D viewer (viz_interactive.py)

`src/viz_interactive.py` generates a **self-contained HTML file** (no server required) with a
fully interactive Three.js 3D scene:

- Orbiting point cloud coloured by scene depth.
- Red camera-position spheres for all 16 frames.
- Coloured OBB wireframes per entity, with per-entity GUI toggle checkboxes.
- Click any OBB centre sphere → HUD displays entity name and 3D centre coordinates.
- Frame projection panel: select any training frame from a dropdown to see OBB corners
  projected onto the actual photograph as a canvas overlay.

**To generate the viewer:**

```python
from src.viz_interactive import build_interactive_viewer
build_interactive_viewer(sparse_pts, sparse_cols, ds.frames, answer,
                         "outputs/scene_viewer.html")
```

Open `outputs/scene_viewer.html` in any browser.

---

## Eval-day workflow

`eval_day_cookbook.py` supports two interchangeable paths for any new entity named on
evaluation day:

**PATH A — DINO-driven (automatic):**

```python
from src.solve_obb import solve_obb
obb, status = solve_obb("new_entity", ["prompt1", "prompt2"],
                        ds.frames, detections, recon["points"])
```

**PATH B — pixel-correspondence fallback (most accurate):**

```python
from src.build_answer import add_entity_via_correspondences
add_entity_via_correspondences(
    entity_name="new_entity",
    pixel_correspondences={"449": (u1, v1), "461": (u2, v2)},
    poses_path="./data/Data/poses.json",
    K=K,
    answer_path="outputs/answer_final.json",
    extent=[0.02, 0.015, 0.01],
)
```

Both paths validate the JSON schema before writing.

---

## Data conventions

- `poses.json` — 4×4 camera-to-world matrices, keyed by frame number (string).
- Image size: 2560 × 1440.
- Intrinsics: `fx = 1477.01, fy = 1480.44, cx = 1298.25, cy = 686.82`. No distortion.
- The data loader inverts c2w to world-to-camera for all downstream geometry (OpenCV
  convention: `P = K [R | t]`).

---

## JSON validation

All output files are validated before writing by `src.build_answer._validate`:

- Top-level type is a list.
- Every entry has `entity` and `obb` keys.
- `obb.center` and `obb.extent` are length-3 lists.
- `obb.rotation` is 3×3 with `det(R) ≈ +1` and `‖R^T R − I‖ ≤ 1e-3`.
- No NaN or Inf values anywhere.

The submitted `answer_final.json` passes all checks.
