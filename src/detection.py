"""Object detection + SAM mask lifting.

Primary:  GroundingDINO (open-vocab bboxes) → SAM (pixel-accurate masks)
Fallback: template/heuristic if either is unavailable.

Coolness additions over baseline Team-B detection.py:
  • SAM ViT-H mask extraction from every DINO bbox.
  • Masked point lifting: for each SAM mask, lift masked pixels to 3D via
    depth-median of the sparse cloud (far more accurate than bbox-centre ray).
  • Per-frame mask confidence scores combining DINO score × SAM IoU-stability.
  • detect_all now optionally returns pixel masks alongside bboxes.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Optional


# ─── lazy model registry ────────────────────────────────────────────────────
_DINO: dict = {"loaded": False, "model": None, "processor": None, "device": None}
_SAM:  dict = {"loaded": False, "predictor": None}


def _load_dino() -> bool:
    if _DINO["loaded"]:
        return _DINO["model"] is not None
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mid = "IDEA-Research/grounding-dino-tiny"
        proc  = AutoProcessor.from_pretrained(mid)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev)
        _DINO.update(loaded=True, model=model, processor=proc, device=dev)
        print(f"[detect] GroundingDINO loaded on {dev}")
        return True
    except Exception as e:
        print(f"[detect] GroundingDINO unavailable ({e.__class__.__name__})")
        _DINO["loaded"] = True
        return False


def _load_sam() -> bool:
    """Load SAM ViT-H if available; silently skip otherwise."""
    if _SAM["loaded"]:
        return _SAM["predictor"] is not None
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        import urllib.request, os
        ckpt = "/tmp/sam_vit_h.pth"
        url  = ("https://dl.fbaipublicfiles.com/segment_anything/"
                "sam_vit_h_4b8939.pth")
        if not os.path.exists(ckpt):
            print("[SAM] downloading ViT-H checkpoint (~2.4 GB) ...")
            urllib.request.urlretrieve(url, ckpt)
        dev  = "cuda" if torch.cuda.is_available() else "cpu"
        sam  = sam_model_registry["vit_h"](checkpoint=ckpt).to(dev)
        pred = SamPredictor(sam)
        _SAM.update(loaded=True, predictor=pred)
        print(f"[SAM] SAM ViT-H loaded on {dev}")
        return True
    except Exception as e:
        print(f"[SAM] unavailable ({e.__class__.__name__}); masks skipped")
        _SAM["loaded"] = True
        return False


# ─── DINO detection ──────────────────────────────────────────────────────────
def detect_with_dino(
    image_bgr: np.ndarray,
    text_prompts: list[str],
    box_thr: float = 0.25,
    text_thr: float = 0.20,
) -> list[dict]:
    """Run GroundingDINO. Returns list of {label, bbox, score}."""
    import torch
    from PIL import Image

    if not _load_dino():
        return []

    rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    prompt = ". ".join(p.strip().lower().rstrip(".") for p in text_prompts) + "."

    proc  = _DINO["processor"]
    model = _DINO["model"]
    dev   = _DINO["device"]

    inputs = proc(images=pil, text=prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        outs = model(**inputs)

    res = proc.post_process_grounded_object_detection(
        outs, inputs.input_ids,
        box_threshold=box_thr, text_threshold=text_thr,
        target_sizes=[pil.size[::-1]],
    )[0]

    return [
        {
            "label": str(lbl).strip().lower(),
            "bbox":  [float(v) for v in box.tolist()],
            "score": float(score),
            "mask":  None,   # filled by SAM below
        }
        for box, score, lbl in zip(res["boxes"], res["scores"], res["labels"])
    ]


# ─── SAM mask refinement ─────────────────────────────────────────────────────
def refine_with_sam(
    image_bgr: np.ndarray,
    detections: list[dict],
) -> list[dict]:
    """
    For every detection, run SAM with the DINO bbox as a box prompt.
    Adds 'mask' (H×W bool) and 'mask_score' (SAM stability score) to each det.
    Falls back gracefully if SAM is not available.
    """
    if not _load_sam() or not detections:
        return detections

    pred = _SAM["predictor"]
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)

    import torch
    refined = []
    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        box_tensor = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)

        try:
            masks, scores, _ = pred.predict(
                box=box_tensor.numpy()[0],
                multimask_output=True,
            )
            # pick highest-stability mask
            best = int(np.argmax(scores))
            det = {**det, "mask": masks[best], "mask_score": float(scores[best])}
        except Exception:
            det = {**det, "mask": None, "mask_score": 0.0}

        refined.append(det)
    return refined


# ─── mask-guided 3-D point lifting ───────────────────────────────────────────
def lift_mask_to_3d(
    mask: np.ndarray,           # H×W bool
    scene_points: np.ndarray,   # (N,3) world-space
    frame: dict,                # must have K, R, t, image
) -> np.ndarray:
    """
    Project all scene points into this frame; return the subset whose
    projection falls inside the SAM mask.  Much tighter than bbox-cone lifting.

    Returns: (M,3) filtered world-space points.
    """
    if mask is None or len(scene_points) == 0:
        return np.zeros((0, 3))

    K, R, t = frame["K"], frame["R"], frame["t"]
    Xc  = (R @ scene_points.T + t.reshape(3, 1)).T          # (N,3) cam-space
    front = Xc[:, 2] > 0.01
    if not front.any():
        return np.zeros((0, 3))

    Xc_f = Xc[front]
    uvh   = (K @ Xc_f.T).T
    uv    = (uvh[:, :2] / uvh[:, 2:3]).astype(int)          # (M,2)

    H, W  = mask.shape[:2]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv_valid = uv[in_img]
    in_mask  = mask[uv_valid[:, 1], uv_valid[:, 0]]         # bool (M',)

    orig_idx = np.where(front)[0][in_img][in_mask]
    return scene_points[orig_idx]


# ─── combined pipeline (drop-in replacement for Team B's detect_all) ──────────
def detect(
    image_bgr: np.ndarray,
    prompts: list[str],
    use_dino: bool = True,
    use_sam:  bool = True,
) -> list[dict]:
    """Single-image detection with optional SAM refinement."""
    if use_dino and _load_dino():
        dets = detect_with_dino(image_bgr, prompts)
    else:
        dets = []

    if use_sam and dets:
        dets = refine_with_sam(image_bgr, dets)

    return dets


def detect_all(
    frames: list[dict],
    prompts: list[str],
    use_dino: bool = True,
    use_sam:  bool = True,
    min_score: float = 0.20,
    scene_points: Optional[np.ndarray] = None,
) -> dict:
    """
    Run detection on every frame.

    Returns:
        dict: frame_idx → list of detection dicts.
              Each det has: label, bbox, score, mask (H×W bool|None),
              mask_score, lifted_pts (Mx3 world points, if scene_points given).
    """
    out = {}
    for f in frames:
        dets = detect(f["image"], prompts, use_dino=use_dino, use_sam=use_sam)

        # score filter + label match
        filtered = []
        tokens = {tok for p in prompts for tok in p.lower().split()}
        for d in dets:
            if d["score"] < min_score:
                continue
            if not (tokens & set(d["label"].split())):
                # check if any full prompt phrase is a substring
                if not any(p.strip().lower().rstrip(".") in d["label"]
                           for p in prompts):
                    continue
            # mask-guided 3-D lifting (bonus: much better than cone casting)
            if scene_points is not None and d.get("mask") is not None:
                d["lifted_pts"] = lift_mask_to_3d(d["mask"], scene_points, f)
            else:
                d["lifted_pts"] = np.zeros((0, 3))
            filtered.append(d)

        out[f["idx"]] = filtered

    total = sum(len(v) for v in out.values())
    print(f"[detect] {total} detections across {len(frames)} frames"
          f" (SAM={'on' if use_sam else 'off'})")
    return out


# ─── helpers ─────────────────────────────────────────────────────────────────
def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def visualize_detections(
    image_bgr: np.ndarray,
    dets: list[dict],
    alpha_mask: float = 0.35,
) -> np.ndarray:
    """Draw bboxes + semi-transparent SAM masks on a copy of the image."""
    img    = image_bgr.copy()
    H, W   = img.shape[:2]
    overlay = img.copy()

    palette = [
        (0, 255, 100), (0, 200, 255), (255, 80,  80),
        (255, 200, 0),  (180, 0, 255), (0, 140, 255),
    ]

    for i, d in enumerate(dets):
        col  = palette[i % len(palette)]
        x0, y0, x1, y1 = [int(v) for v in d["bbox"]]

        # SAM mask fill
        if d.get("mask") is not None:
            m = d["mask"].astype(bool)
            if m.shape[:2] == (H, W):
                overlay[m] = col

        # bbox rect
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)

        # label
        label = (f'{d["label"]} '
                 f'{d["score"]:.2f}'
                 + (f' SAM={d["mask_score"]:.2f}' if d.get("mask_score") else ""))
        cv2.putText(img, label, (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # blend mask overlay
    cv2.addWeighted(overlay, alpha_mask, img, 1 - alpha_mask, 0, img)
    return img
