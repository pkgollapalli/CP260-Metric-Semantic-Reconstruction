"""Data loading: images + poses.json + intrinsics.

poses.json expected format (flexible — handles common variants):
  { "319": {"R": [[..3x3..]], "t": [..3..], "K": [[..3x3..]]},  ... }
  OR { "319": {"transform_matrix": [[..4x4..]], "K": ...},     ... }
  OR { "319": {"qvec":[w,x,y,z], "tvec":[..]},                  ... }

Convention used everywhere downstream: P = K [R | t], world->camera (OpenCV).
If poses.json stores camera->world (NeRF/Blender convention), we invert.
"""
import json
import os
import re
from pathlib import Path
import numpy as np
import cv2


# ---------- pose conversion helpers ----------
def qvec_to_R(q):
    """Quaternion (w,x,y,z) -> rotation matrix (Hamilton convention)."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def R_to_qvec(R):
    """Rotation matrix -> quaternion (w,x,y,z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def parse_pose_entry(entry, default_K=None, assume_w2c=True):
    """Parse one pose entry into (K, R, t) world->camera."""
    K = None
    if "K" in entry:
        K = np.array(entry["K"], dtype=np.float64)
    elif "intrinsics" in entry:
        K = np.array(entry["intrinsics"], dtype=np.float64)
    elif default_K is not None:
        K = default_K.copy()

    if "transform_matrix" in entry or "T" in entry or "pose" in entry:
        T = np.array(entry.get("transform_matrix",
                               entry.get("T", entry.get("pose"))),
                     dtype=np.float64)
        if T.shape == (4, 4):
            R, t = T[:3, :3], T[:3, 3]
        else:
            R, t = T[:3, :3], T[:3, 3]
        if not assume_w2c:
            # camera->world given, invert to world->camera
            R = R.T
            t = -R @ t
    elif "R" in entry and "t" in entry:
        R = np.array(entry["R"], dtype=np.float64)
        t = np.array(entry["t"], dtype=np.float64).reshape(3)
        if not assume_w2c:
            R = R.T
            t = -R @ t
    elif "qvec" in entry and "tvec" in entry:
        # COLMAP-style: stored as world->camera by default
        R = qvec_to_R(entry["qvec"])
        t = np.array(entry["tvec"], dtype=np.float64).reshape(3)
    else:
        raise ValueError(f"Unrecognized pose entry keys: {list(entry.keys())}")

    return K, R, t


# ---------- main loader ----------
class DesktopDataset:
    def __init__(self, data_dir, poses_file=None, default_K=None, w2c=True):
        self.data_dir = Path(data_dir)
        if poses_file is None:
            # find poses.json
            cands = list(self.data_dir.glob("*poses*.json"))
            if not cands:
                cands = list(self.data_dir.parent.glob("*poses*.json"))
            poses_file = cands[0] if cands else None
        if poses_file is None:
            raise FileNotFoundError("poses.json not found")
        with open(poses_file) as f:
            self.raw_poses = json.load(f)

        self.default_K = np.array(default_K, dtype=np.float64) if default_K is not None else None
        self.w2c = w2c
        self.frames = self._build_frames()
        if not self.frames:
            raise RuntimeError("No frames matched between images and poses.")
        H, W = self.frames[0]["image"].shape[:2]
        self.H, self.W = H, W

    def _build_frames(self):
        frames = []
        # collect available images in dir
        img_paths = {}
        for p in sorted(self.data_dir.glob("frame_*.png")):
            m = re.search(r"(\d+)", p.stem)
            if m:
                img_paths[int(m.group(1))] = p

        # poses keyed by frame number (string or int)
        for key, entry in self.raw_poses.items():
            try:
                idx = int(key)
            except ValueError:
                m = re.search(r"(\d+)", str(key))
                if not m:
                    continue
                idx = int(m.group(1))
            if idx not in img_paths:
                continue
            try:
                K, R, t = parse_pose_entry(entry, self.default_K, assume_w2c=self.w2c)
            except Exception as e:
                print(f"[skip] frame {idx}: {e}")
                continue
            img = cv2.imread(str(img_paths[idx]))  # BGR
            if img is None:
                continue
            P = K @ np.hstack([R, t.reshape(3, 1)])
            C = -R.T @ t  # camera center in world
            frames.append({
                "idx": idx,
                "path": str(img_paths[idx]),
                "image": img,
                "K": K, "R": R, "t": t.reshape(3),
                "P": P, "C": C,
            })
        frames.sort(key=lambda f: f["idx"])
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


if __name__ == "__main__":
    # smoke test
    import sys
    if len(sys.argv) > 1:
        ds = DesktopDataset(sys.argv[1])
        print(f"Loaded {len(ds)} frames at {ds.H}x{ds.W}")
        for f in ds.frames[:3]:
            print(f"  idx={f['idx']} C={f['C'].round(3)}")