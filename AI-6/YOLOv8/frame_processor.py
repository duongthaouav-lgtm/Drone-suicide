"""
frame_processor.py - Per-frame inference, tracking, and annotation logic
Optimised for OpenVINO on Pi 5: minimal overhead, no redundant work.
"""
import time
from collections import defaultdict

import cv2
import numpy as np

import config as cfg
import perf_monitor
import utils as tu
from targeting_overlay import draw_targeting_overlay, reset_lock, is_locked, get_locked_tid
from utils import (
    KALMAN_AVAILABLE,
    adaptive_confidence_adjustment,
)

# ── Import hardware flags once at module level (not inside every frame) ───────
from hardware import IS_RASPBERRY_PI5, HAS_HAILO_NPU

# Max detections cap — computed once
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    _MAX_DET = 10
elif IS_RASPBERRY_PI5 and HAS_HAILO_NPU:
    _MAX_DET = 20
else:
    _MAX_DET = 50

# Whether to use tracker (OpenVINO is incompatible with ByteTrack Kalman)
_USE_TRACKER = (
    cfg.ENABLE_TRACKING
    and cfg.TRACKER_TYPE in ("bytetrack", "botsort", "hybrid")
    and not cfg.USE_OV
)
_TRACKER_YAML = "botsort.yaml" if cfg.TRACKER_TYPE == "botsort" else "bytetrack.yaml"

# =====================================================
# MUTABLE TRACKING STATE
# =====================================================
tracking_history: dict = {}
last_annotated_frame    = None
skip_frame_counter: int = 0

tracking_stability: dict = {}
last_frame_results: dict | None = None


def reset_state():
    """Clear all per-video state. Call before each new video."""
    global tracking_history 
    global last_annotated_frame, skip_frame_counter
    global tracking_stability, last_frame_results

    tracking_history.clear()
    tracking_stability.clear()
    last_frame_results  = None
    last_annotated_frame = None
    skip_frame_counter  = 0

    tu.reset_state()
    perf_monitor.reset()
    reset_lock()


# =====================================================
# MAIN FRAME PROCESSING
# =====================================================
def process_frame(model, device: str, frame, frame_count: int = 0,
                  total_frames: int = 0, fps: float = 0.0,
                  logger=None) -> tuple:
    global last_annotated_frame

    frame_start = time.time()

    # ── Frame skip — reuse last annotated frame ───────────────────────────────
    if (cfg.USE_FRAME_SKIP
            and not is_locked()
            and frame_count % cfg.FRAME_SKIP_N != 0
            and last_annotated_frame is not None):
        return last_annotated_frame, _empty_stats(frame_start)

    # ── Confidence threshold ──────────────────────────────────────────────────
    conf = cfg.CONF_THRESH
    if cfg.USE_ADAPTIVE_CONF and not cfg.USE_OV:
        # Adaptive conf disabled for OpenVINO — fixed graph, no benefit
        conf = adaptive_confidence_adjustment(tu.detection_quality_history)
        if _USE_TRACKER:
            conf = max(conf, cfg.TRACK_CONF_THRESH)

    # ── Image size — OpenVINO has fixed input, skip adaptive logic ─────────────
    imgsz = cfg.IMGSZ  # OpenVINO ignores this anyway (baked in at export)

    # ── Inference ────────────────────────────────────────────────────────────
    inf_start = time.time()

    if _USE_TRACKER:
        results = model.track(
            frame,
            conf=conf,
            iou=cfg.TRACK_IOU_THRESH,
            imgsz=imgsz,
            verbose=False,
            device=device,
            half=False,   # OpenVINO handles precision internally
            persist=True,
            tracker=_TRACKER_YAML,
            show=False,
            agnostic_nms=False,
            max_det=_MAX_DET,
            retina_masks=False,
            stream=False,
        )
    else:
        results = model.predict(
            frame,
            conf=conf,
            iou=cfg.IOU_THRESH,
            imgsz=imgsz,
            verbose=False,
            device=device,
            half=False if cfg.USE_OV else cfg.USE_HALF,   # INT8 conflicts with FP16
            show=False,
            agnostic_nms=False,
            max_det=_MAX_DET,
            retina_masks=False,
            stream=False,
        )

    inference_time = time.time() - inf_start
    post_start = time.time()

    # ── Unpack result (YOLO returns a list, always one element here) ──────────
    names      = model.names
    r          = results[0]
    annotated  = frame.copy()

    detection_count = 0
    track_count     = 0
    class_counts: dict[str, int] = defaultdict(int)
    track_ids_set:  set[int]     = set()
    targeting_dets: list         = []

    locked_tid = get_locked_tid() if is_locked() else None

    boxes = r.boxes

    if boxes is not None and len(boxes) > 0:
        # ── Pull tensors once ─────────────────────────────────────────────────
        boxes_data = boxes.xyxy.cpu().numpy()
        confs_arr  = boxes.conf.cpu().numpy()
        clss_arr   = boxes.cls.cpu().numpy().astype(int)
        track_ids  = (boxes.id.cpu().numpy().astype(int)
                      if boxes.id is not None else None)

        if track_ids is not None:
            track_count = len(track_ids)

        # ── Filter NaN/Inf (can occur with OpenVINO output) ─────────────────────────
        valid = np.all(np.isfinite(boxes_data), axis=1)
        if not valid.all():
            boxes_data = boxes_data[valid]
            confs_arr  = confs_arr[valid]
            clss_arr   = clss_arr[valid]
            if track_ids is not None:
                track_ids = track_ids[valid[:len(track_ids)]]


        # ── Per-box loop ──────────────────────────────────────────────────────
        for i, (box, conf_val, cls) in enumerate(zip(boxes_data, confs_arr, clss_arr)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = names[cls]
            tid   = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else None

            # Skip non-locked tracks when locked
            if locked_tid is not None and tid != locked_tid:
                continue

            # Draw box
            color     = _track_color(tid)
            thickness = 2 if tid is None else 3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label
            lbl = f"ID:{tid} {label} {conf_val:.2f}" if tid is not None else f"{label} {conf_val:.2f}"
            _draw_label(annotated, lbl, x1, y1, color)

            detection_count += 1
            class_counts[label] += 1
            if tid is not None:
                track_ids_set.add(tid)

            # Build targeting list in same loop (no second pass needed)
            targeting_dets.append({
                'box':  (x1, y1, x2, y2),
                'label': label,
                'conf':  float(conf_val),
                'tid':   tid,
            })

    raw_det_count = detection_count

    # ── HUD ──────────────────────────────────────────────────────────────────
    _draw_hud(annotated, frame_count, total_frames, detection_count,
              raw_det_count, track_count, fps, tu.current_adaptive_conf, device)

    # ── Targeting overlay ─────────────────────────────────────────────────────
    targeting_dets.sort(key=lambda d: d['conf'], reverse=True)
    if locked_tid is not None:
        locked_dets = [d for d in targeting_dets if d.get('tid') == locked_tid]
        draw_targeting_overlay(annotated, locked_dets if locked_dets else targeting_dets[:1])
    else:
        draw_targeting_overlay(annotated, targeting_dets)

    # ── Logging ───────────────────────────────────────────────────────────────
    if (logger and cfg.LOG_EVERY_N_FRAMES > 0
            and frame_count % cfg.LOG_EVERY_N_FRAMES == 0):
        classes_str = ", ".join(f"{k}:{v}" for k, v in class_counts.items())
        logger.info(
            f"Frame {frame_count}/{total_frames}: Det={detection_count}, "
            f"Tracks={track_count}, FPS={fps:.1f}"
            + (f" | Classes: {classes_str}" if classes_str else "")
        )

    # ── Adaptive conf quality (skip for OpenVINO) ─────────────────────────────────
    if cfg.USE_ADAPTIVE_CONF and not cfg.USE_OV:
        quality = 1.0 if detection_count > 0 else 0.0
        tu.detection_quality_history.append(quality)
        if len(tu.detection_quality_history) > 30:
            tu.detection_quality_history.pop(0)

    # ── Perf + memory ─────────────────────────────────────────────────────────
    postprocess_time = time.time() - post_start
    total_time       = time.time() - frame_start
    perf_monitor.update(inference_time, 0, postprocess_time)

    if (cfg.ENABLE_MEMORY_OPTIMIZATION
            and frame_count > 0
            and frame_count % cfg.MEMORY_CLEANUP_INTERVAL == 0):
        perf_monitor.cleanup_memory()

    # Cache for frame skip (no copy — annotated is already a new array from frame.copy())
    last_annotated_frame = annotated

    return annotated, {
        'detection_count': detection_count,
        'track_count':     track_count,
        'class_counts':    dict(class_counts),
        'track_ids':       list(track_ids_set),
        'inference_time_ms': inference_time * 1000,
        'total_time_ms':     total_time * 1000,
    }


# =====================================================
# HELPERS
# =====================================================
def _empty_stats(frame_start: float) -> dict:
    return {
        'detection_count': 0,
        'track_count':     0,
        'class_counts':    {},
        'track_ids':       [],
        'inference_time_ms': 0,
        'total_time_ms':   (time.time() - frame_start) * 1000,
    }


def _track_color(tid) -> tuple:
    if tid is None:
        return (0, 255, 0)
    c = tid % 255
    return (
        int(255 * np.sin(c * 0.1) ** 2),
        int(255 * np.sin(c * 0.1 + 2) ** 2),
        int(255 * np.sin(c * 0.1 + 4) ** 2),
    )


def _draw_label(img, text: str, x1: int, y1: int, color: tuple):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, max(y1 - 5, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _draw_hud(img, frame_count, total_frames, det_count, raw_count,
              track_count, fps, adaptive_conf, device):
    import torch as _torch
    tracker_tag = cfg.TRACKER_TYPE.upper() if cfg.ENABLE_TRACKING and not cfg.USE_OV else "OV"
    device_tag  = "GPU" if _torch.cuda.is_available() else "CPU"

    info = (f"Frame: {frame_count}/{total_frames}" if total_frames > 0
            else "Frame: Live")
    info += f" | Det: {det_count}"
    if cfg.ENABLE_TRACKING and track_count > 0:
        info += f" | Tracks: {track_count} [{tracker_tag}]"
    if cfg.SHOW_FPS and fps > 0:
        info += f" | FPS: {fps:.1f}"
    info += f" | {device_tag}"

    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (10, 10), (20 + tw, 40), (0, 0, 0), -1)
    cv2.putText(img, info, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)