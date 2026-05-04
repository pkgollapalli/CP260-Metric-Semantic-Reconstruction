"""
Build the final answer JSON.

This script produces ``outputs/answer_final.json`` deterministically from
manually verified pixel correspondences and the camera poses. It can be run
independently (no GPU, no DINO required) to reproduce the submission.

The pipeline is:
  1) Load camera intrinsics and poses.
  2) For each baseline entity (VGA, Ethernet, Power), use the pixel
     correspondences listed below as ``MEASUREMENTS``.
  3) Triangulate the 3D centre with multi-pair median DLT.
  4) Project the centre onto the I/O-panel plane.
  5) Use IEC connector standard extents + GT VGA rotation.
  6) Validate JSON schema (det(R) = +1, R^T R = I, no NaN, etc.).
  7) Write ``outputs/answer_final.json``.

The same module exposes ``add_entity_via_correspondences`` so that on
evaluation day, a new entity can be added by reading two or three pixel
locations from any frame and calling this helper.
"""
from __future__ import annotations

import json
import os
import argparse
import numpy as np

from src.solve_obb import (
    GT_VGA_CENTER, GT_VGA_EXTENT, GT_VGA_ROTATION,
    PHYS_EXTENT,
    solve_obb_from_correspondences,
)


# ---------------------------------------------------------------------------
# Pixel correspondences obtained from the released dataset.
# These were read manually from the calibration frames using the methodology
# documented in the report (3-pair median DLT triangulation).
# ---------------------------------------------------------------------------
MEASUREMENTS = {
    "ethernet_socket": {
        "365": (1550, 550),
        "449": (1660, 525),
        "461": (1899, 516),
    },
    "power_socket": {
        "400": (1375, 900),
        "449": (1650, 925),
        "461": (1882, 897),
    },
}


def build_answer(poses_path, K, output_path):
    with open(poses_path) as f:
        poses = json.load(f)

    answer = []

    # 1) VGA -- ground truth, copied verbatim.
    answer.append({
        "entity": "vga_socket",
        "obb": {
            "center":   GT_VGA_CENTER.tolist(),
            "extent":   GT_VGA_EXTENT.tolist(),
            "rotation": GT_VGA_ROTATION.tolist(),
        }
    })

    # 2) Ethernet
    eth_obb = solve_obb_from_correspondences(
        pixel_correspondences=MEASUREMENTS["ethernet_socket"],
        poses=poses, K=K,
        extent=PHYS_EXTENT["ethernet_socket"],
    )
    answer.append({"entity": "ethernet_socket", "obb": eth_obb})

    # 3) Power
    pow_obb = solve_obb_from_correspondences(
        pixel_correspondences=MEASUREMENTS["power_socket"],
        poses=poses, K=K,
        extent=PHYS_EXTENT["power_socket"],
    )
    answer.append({"entity": "power_socket", "obb": pow_obb})

    # 4) Validate
    _validate(answer)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(answer, f, indent=2)
    print(f"Wrote {output_path}")
    print(json.dumps(answer, indent=2))
    return answer


def _validate(answer):
    """Strict schema validation. Raises if anything is malformed."""
    assert isinstance(answer, list), "Top level must be a list"
    for i, entry in enumerate(answer):
        assert "entity" in entry, f"[{i}] missing 'entity'"
        assert "obb" in entry, f"[{i}] missing 'obb'"
        obb = entry["obb"]
        for key in ("center", "extent", "rotation"):
            assert key in obb, f"[{i}] obb missing '{key}'"
        c, e, R = obb["center"], obb["extent"], obb["rotation"]
        assert len(c) == 3 and len(e) == 3, f"[{i}] center/extent must be length-3"
        assert len(R) == 3 and all(len(row) == 3 for row in R), \
            f"[{i}] rotation must be 3x3"
        Rm = np.array(R)
        det = np.linalg.det(Rm)
        orth = np.linalg.norm(Rm.T @ Rm - np.eye(3))
        assert abs(det - 1.0) < 1e-3, \
            f"[{i}] {entry['entity']} det(R) = {det}, must be +1"
        assert orth < 1e-3, \
            f"[{i}] {entry['entity']} R not orthonormal"
        flat = np.array(c + e + sum(R, []))
        assert np.isfinite(flat).all(), \
            f"[{i}] {entry['entity']} contains NaN/Inf"
    print(f"  validated {len(answer)} entries")


def add_entity_via_correspondences(entity_name, pixel_correspondences,
                                     poses_path, K, answer_path,
                                     extent=None):
    """Add a new entity using manually measured pixel correspondences.

    Use this on evaluation day for any entity that DINO does not detect
    reliably or that needs sub-pixel accuracy.
    """
    with open(poses_path) as f:
        poses = json.load(f)
    with open(answer_path) as f:
        answer = json.load(f)

    if extent is None:
        extent = PHYS_EXTENT.get(entity_name, [0.02, 0.015, 0.01])

    obb = solve_obb_from_correspondences(
        pixel_correspondences=pixel_correspondences,
        poses=poses, K=K, extent=extent,
    )
    answer = [e for e in answer if e["entity"] != entity_name]
    answer.append({"entity": entity_name, "obb": obb})
    _validate(answer)
    with open(answer_path, "w") as f:
        json.dump(answer, f, indent=2)
    print(f"Appended {entity_name} -> {answer_path}")
    return obb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--K", default="1477.01,1480.44,1298.25,686.82")
    ap.add_argument("--output", default="outputs/answer_final.json")
    args = ap.parse_args()

    fx, fy, cx, cy = [float(x) for x in args.K.split(",")]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    build_answer(args.poses, K, args.output)


if __name__ == "__main__":
    main()
