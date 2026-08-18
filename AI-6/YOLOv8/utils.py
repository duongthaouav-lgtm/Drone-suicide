"""
tracker_utils.py - Tracking helper utilities

Includes:
  - Kalman Filter creation / update
  - IoU calculation
  - Hungarian matching
  - Adaptive confidence adjustment
  - Adaptive image-size adjustment
  - Occlusion detection
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

import config as cfg

# =====================================================
# OPTIONAL DEPENDENCY: filterpy (Kalman Filter)
# =====================================================
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print(
        "Warning: filterpy not installed. Kalman Filter disabled. "
        "Install with: pip install filterpy"
    )

# =====================================================
# MODULE-LEVEL STATE (initialised / reset by caller)
# =====================================================
current_adaptive_conf: float = cfg.CONF_THRESH
detection_quality_history: list[float] = []
adaptive_imgsz_history: list[int] = []
scene_complexity: float = 0.5


def reset_state():
    """Reset all module-level mutable state (call when starting a new video)."""
    global current_adaptive_conf, detection_quality_history
    global adaptive_imgsz_history, scene_complexity
    current_adaptive_conf = cfg.CONF_THRESH
    detection_quality_history.clear()
    adaptive_imgsz_history.clear()
    scene_complexity = 0.5

# =====================================================
# IoU
# =====================================================
def calculate_iou(box1, box2) -> float:
    """Return Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    ix1 = max(x1_min, x2_min)
    iy1 = max(y1_min, y2_min)
    ix2 = min(x1_max, x2_max)
    iy2 = min(y1_max, y2_max)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# =====================================================
# ADAPTIVE CONFIDENCE
# =====================================================
def adaptive_confidence_adjustment(quality_history: list[float]) -> float:
    """
    Adjust the confidence threshold based on recent detection quality.
    A low detection rate drives the threshold down; high quality drives it up.
    """
    global current_adaptive_conf

    if not cfg.USE_ADAPTIVE_CONF or len(quality_history) < 2:
        return current_adaptive_conf

    recent = quality_history[-8:]
    avg_quality = float(np.mean(recent))
    det_rate = sum(1 for q in recent if q > 0) / len(recent)

    if det_rate < 0.60:
        current_adaptive_conf = max(
            cfg.ADAPTIVE_CONF_MIN,
            current_adaptive_conf - cfg.ADAPTIVE_CONF_STEP * 2.5,
        )
    elif det_rate < 0.70:
        current_adaptive_conf = max(
            cfg.ADAPTIVE_CONF_MIN,
            current_adaptive_conf - cfg.ADAPTIVE_CONF_STEP,
        )
    elif avg_quality < 0.5:
        current_adaptive_conf = max(
            cfg.ADAPTIVE_CONF_MIN,
            current_adaptive_conf - cfg.ADAPTIVE_CONF_STEP * 0.5,
        )
    elif avg_quality > 0.85:
        current_adaptive_conf = min(
            cfg.ADAPTIVE_CONF_MAX,
            current_adaptive_conf + cfg.ADAPTIVE_CONF_STEP,
        )

    return current_adaptive_conf
