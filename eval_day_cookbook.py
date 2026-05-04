"""
EVAL-DAY COOKBOOK
=================

Run this in Colab (or locally) on the day of evaluation.
The instructor will name 1-2 additional entities; this script computes the OBB
for each, appends to the existing answer JSON, and validates the schema before
writing the file.

There are TWO paths, in order of preference:

  PATH A -- DINO-driven (no human input needed). Use when the entity is one
  of the typical I/O panel ports (USB, HDMI, audio, ...). The detector
  usually finds it; the ray-cone first-surface solver returns a clean centre.

  PATH B -- pixel-correspondence fallback. Use when DINO is unreliable.
  Read the entity's pixel centre in any TWO or THREE clear frames; the
  module triangulates from those correspondences directly. This is what was
  used for the three baseline entities and matches the projection within
  ~2 px in every checked view.

Both paths produce the same JSON schema.
"""

# ============================================================
# 0. Setup -- run once at the start of the eval session
# ============================================================
import sys, os, json
import numpy as np

sys.path.insert(0, '/content/cp260_final')   # adapt to actual path
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

from src.data_loader import DesktopDataset
from src.reconstruction import reconstruct
from src.detection import detect_all
from src.solve_obb import (
    solve_obb,
    solve_obb_from_correspondences,
    DEFAULT_PROMPTS,
    PHYS_EXTENT,
)
from src.build_answer import (
    build_answer,
    add_entity_via_correspondences,
    _validate,
)

# Paths -- adapt to the eval-day environment
DATA_DIR    = "/content/drive/MyDrive/cp260/Data/Data"
POSES_PATH  = "/content/drive/MyDrive/cp260/Data/Data/poses.json"
ANSWER_PATH = "/content/outputs/answer_final.json"

K = np.array([[1477.01, 0, 1298.25],
              [0, 1480.44, 686.82],
              [0, 0, 1]])

# Build the baseline answer (VGA, ETH, POW) -- already in the repo, but
# rerun here so we are using the same code path as the instructor's grader.
build_answer(POSES_PATH, K, ANSWER_PATH)


# ============================================================
# 1. Load data + run reconstruction (PATH A only -- skip if using PATH B)
# ============================================================
ds = DesktopDataset(DATA_DIR, POSES_PATH, default_K=K, w2c=False)
recon = reconstruct(ds.frames, n_features=4000)
all_pts = recon["points"]

# Multi-prompt DINO sweep -- captures a wide range of port labels with one pass
all_prompts = []
for kw_list in DEFAULT_PROMPTS.values():
    all_prompts.extend(kw_list)
all_prompts = list(dict.fromkeys(all_prompts))
detections = detect_all(ds.frames, all_prompts, use_dino=True, min_score=0.20)
print(f"detections collected: {sum(len(v) for v in detections.values())}")


# ============================================================
# 2. PATH A -- DINO-driven for one or two new entities
# ============================================================
def add_via_dino(entity_name, prompt_keywords):
    obb, status = solve_obb(entity_name, prompt_keywords,
                              ds.frames, detections, all_pts)
    print(f"[{entity_name}] {status}")
    if obb is None:
        print(f"  PATH A failed for {entity_name} -- fall back to PATH B")
        return None

    with open(ANSWER_PATH) as f:
        answer = json.load(f)
    answer = [e for e in answer if e["entity"] != entity_name]
    answer.append({"entity": entity_name, "obb": obb})
    _validate(answer)
    with open(ANSWER_PATH, "w") as f:
        json.dump(answer, f, indent=2)
    return obb


# Examples -- replace with whatever the instructor asks for
add_via_dino("usb_port",   DEFAULT_PROMPTS["usb_port"])
add_via_dino("hdmi_port",  DEFAULT_PROMPTS["hdmi_port"])
add_via_dino("audio_jack", DEFAULT_PROMPTS["audio_jack"])


# ============================================================
# 3. PATH B -- pixel-correspondence fallback
# ============================================================
# If PATH A fails, identify the entity in two or three clear frames using a
# grid overlay (see ``notebooks/measure_pixels.ipynb``) and call this:

# Example: usb_port via measured pixels in frames 449 and 461
add_entity_via_correspondences(
    entity_name="usb_port",
    pixel_correspondences={"449": (1660, 310), "461": (1910, 310)},
    poses_path=POSES_PATH,
    K=K,
    answer_path=ANSWER_PATH,
    extent=PHYS_EXTENT["usb_port"],
)

add_entity_via_correspondences(
    entity_name="hdmi_port",
    pixel_correspondences={"449": (1640, 405), "461": (1885, 415)},
    poses_path=POSES_PATH,
    K=K,
    answer_path=ANSWER_PATH,
    extent=PHYS_EXTENT["hdmi_port"],
)


# ============================================================
# 4. Final visual sanity check before submitting
# ============================================================
import cv2, matplotlib.pyplot as plt

with open(POSES_PATH) as f:
    poses = json.load(f)
with open(ANSWER_PATH) as f:
    answer = json.load(f)


def project(X, frame_key):
    T = np.array(poses[frame_key])
    R = T[:3, :3].T
    t = -T[:3, :3].T @ T[:3, 3]
    Xc = R @ X + t
    uvh = K @ Xc
    return uvh[:2] / uvh[2]


colors = ['red', 'lime', 'cyan', 'orange', 'magenta', 'yellow',
          'white', 'pink', 'lightblue']

for fkey in ("449", "461"):
    img = cv2.imread(f"{DATA_DIR}/frame_000{fkey}.png")
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i, e in enumerate(answer):
        uv = project(np.array(e["obb"]["center"]), fkey)
        if 0 < uv[0] < 2560 and 0 < uv[1] < 1440:
            c = colors[i % len(colors)]
            ax.plot(*uv, "+", color=c, markersize=22, mew=3)
            ax.text(uv[0] + 12, uv[1], e["entity"],
                    color=c, fontsize=9, fontweight="bold")
    ax.set_title(f"frame {fkey}", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Final schema validation -- run this LAST before submitting
# ============================================================
with open(ANSWER_PATH) as f:
    answer = json.load(f)
_validate(answer)
print(f"OK: {len(answer)} entities, schema valid, ready to submit.")

# Download the file
from google.colab import files
files.download(ANSWER_PATH)
