

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional, Deque


# ─────────────────────────────────────────────
#  Tunable constants  (adjust to your drone)
# ─────────────────────────────────────────────

# Speed limits  (m/s)
MAX_SPEED   = 6.5    # cruise / acceleration ceiling
# CREEP_SPEED = 0.8    # slow final approach
# STOP_SPEED  = 0.0

# Distance thresholds  (metres, estimated)
# DIST_FAR    = 10.0   # beyond this → full speed
# DIST_CLOSE  =  5.0   # inside this → start slowing
# DIST_DEAD   =  1.0   # inside this → stop

# Proportional-gain for lateral steering  (tune per airframe)
KP_LATERAL  = 0.004  # pixel-error → m/s  lateral correction
KP_VERTICAL = 0.004

# ── PD Yaw controller ─────────────────────────────────────────────────
# P gain : how hard to yaw toward the current position error
# D gain : how hard to yaw toward where the target is GOING (prediction)
# MAX_YAW_RATE : physical ceiling for yaw command (deg/s)
KP_YAW          = 0.03   # deg/s per pixel of position error
KD_YAW          = 0.01   # deg/s per pixel/s of target velocity
MAX_YAW_RATE    = 25.0   # deg/s ceiling
YAW_DEADZONE_PX = 20     # pixels — no yaw command inside this band

# Velocity estimator: smoothing window for target-motion estimation
VEL_SMOOTH_FRAMES = 5    # number of recent frames to average velocity over

# Acceleration ramp  (m/s²) — limits how fast speed changes
ACCEL_RAMP  = 2.0    # positive: can accelerate up to this per second
DECEL_RAMP  = 5.0    # negative: can brake up to this per second

# Camera / optics calibration
# Real-world reference: when a ~1.8 m tall person stands X metres away,
# their bounding-box height is Y pixels on a 640×360 frame.
# Adjust REFERENCE_HEIGHT_M and FOCAL_LENGTH_PX to match your camera.
REFERENCE_HEIGHT_M  = 0.5    # assumed target real height (metres)
FOCAL_LENGTH_PX     = 400.0  # approximate focal length in pixels


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class GuidanceCommand:
    """Output command produced each frame."""
    vx:           float = 0.0   # forward  (+) / backward (-)  m/s
    vy:           float = 0.0   # right    (+) / left     (-)  m/s  (lateral)
    vz:           float = 0.0   # up       (+) / down     (-)  m/s  (altitude hold)
    yaw_rate:     float = 0.0   # clockwise (+) / counter-clockwise (-)  deg/s
    speed:        float = 0.0   # scalar forward speed command
    est_dist:     float = 0.0   # estimated distance to target (m)
    phase:        str   = "IDLE"  # FAR | MID | CLOSE | DEAD | LOST | IDLE
    err_x:        float = 0.0   # horizontal pixel error (+ = target right of centre)
    err_y:        float = 0.0   # vertical   pixel error (+ = target below  centre)
    target_vel_x: float = 0.0   # target horizontal velocity in frame (px/s), + = moving right
    target_vel_y: float = 0.0   # target vertical   velocity in frame (px/s), + = moving down
    yaw_p:        float = 0.0   # debug: P contribution to yaw
    yaw_d:        float = 0.0   # debug: D contribution to yaw (predictive)


# ─────────────────────────────────────────────
#  Main guidance class
# ─────────────────────────────────────────────

class DroneGuidance:
    def __init__(
        self,
        frame_w: int = 640,
        frame_h: int = 360,
        ref_height_m:    float = REFERENCE_HEIGHT_M,
        focal_length_px: float = FOCAL_LENGTH_PX,
    ):
        self.frame_w         = frame_w
        self.frame_h         = frame_h
        self.ref_height_m    = ref_height_m
        self.focal_length_px = focal_length_px

        self._current_speed  = 0.0

        # ── Target velocity estimator ──────────────────────────────────
        # Stores recent (err_x, err_y, timestamp) to compute smooth velocity
        self._err_history: Deque[Tuple[float, float, float]] = deque(
            maxlen=VEL_SMOOTH_FRAMES
        )
        self._prev_err_x: Optional[float] = None
        self._prev_err_y: Optional[float] = None

    # ------------------------------------------------------------------
    def compute(
        self,
        bbox:    Optional[Tuple[int, int, int, int]],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        dt:      float = 0.033,
    ) -> GuidanceCommand:
        cmd = GuidanceCommand()

        if bbox is None:
            cmd.phase = "LOST"
            self._current_speed = 0.0
            self._err_history.clear()
            self._prev_err_x = None
            self._prev_err_y = None
            return cmd

        fw = frame_w or self.frame_w
        fh = frame_h or self.frame_h

        x, y, w, h = bbox

        # ── 1.  Frame-centre error  ────────────────────────────────────
        cx = x + w / 2
        cy = y + h / 2

        err_x = cx - (fw / 2)   # + = target right of centre
        err_y = cy - (fh / 2)   # + = target below  centre

        cmd.err_x = err_x
        cmd.err_y = err_y

        # ── 2.  Target velocity estimation (smoothed)  ─────────────────
        # Push current error + timestamp into history window
        import time as _time
        now = _time.perf_counter()
        self._err_history.append((err_x, err_y, now))

        vel_x, vel_y = 0.0, 0.0
        if len(self._err_history) >= 2:
            oldest = self._err_history[0]
            newest = self._err_history[-1]
            elapsed = newest[2] - oldest[2]
            if elapsed > 1e-4:                        # avoid division by zero
                vel_x = (newest[0] - oldest[0]) / elapsed   # px/s
                vel_y = (newest[1] - oldest[1]) / elapsed

        cmd.target_vel_x = vel_x   # + = target moving RIGHT  in frame
        cmd.target_vel_y = vel_y   # + = target moving DOWN   in frame

        # ── 3.  Estimate distance from bbox height  ────────────────────
        # if h > 0:
        #     est_dist = (self.ref_height_m * self.focal_length_px) / h
        # else:
        #     est_dist = DIST_FAR

        # cmd.est_dist = est_dist

        # ── 4.  Speed scheduling  ──────────────────────────────────────
        target_speed = MAX_SPEED
        # cmd.phase    = self._phase_name(est_dist)

        self._current_speed = self._ramp_speed(
            self._current_speed, target_speed, dt
        )
        cmd.speed = self._current_speed
        cmd.vx    = self._current_speed

        # ── 5.  PD Yaw controller  ─────────────────────────────────────
        # Apply dead-zone so tiny errors don't cause jitter
        effective_err_x = err_x if abs(err_x) > YAW_DEADZONE_PX else 0.0

        yaw_p = KP_YAW * effective_err_x          # position error term
        yaw_d = KD_YAW * vel_x                    # velocity / prediction term

        raw_yaw = yaw_p + yaw_d
        # Clamp to physical limit
        cmd.yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, raw_yaw))
        cmd.yaw_p    = yaw_p
        cmd.yaw_d    = yaw_d

        # ── 6.  Lateral & vertical corrections  ───────────────────────
        cmd.vy = KP_LATERAL  * err_x    # sideslip to stay aligned
        cmd.vz = -KP_VERTICAL * err_y   # altitude correction (invert Y-axis)

        cmd.vy = max(-1.0, min(1.0, cmd.vy))
        cmd.vz = max(-1.5, min(1.5, cmd.vz))

        return cmd

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    # def _speed_schedule(self, dist: float) -> float:
    #     """Map estimated distance → desired forward speed (m/s)."""
    #     if dist <= DIST_DEAD:
    #         return STOP_SPEED

    #     if dist <= DIST_CLOSE:
    #         # Linear ramp: STOP at DIST_DEAD, CREEP at DIST_CLOSE
    #         t = (dist - DIST_DEAD) / (DIST_CLOSE - DIST_DEAD)
    #         return CREEP_SPEED * t

    #     if dist <= DIST_FAR:
    #         # Linear ramp: CREEP at DIST_CLOSE, MAX at DIST_FAR
    #         t = (dist - DIST_CLOSE) / (DIST_FAR - DIST_CLOSE)
    #         return CREEP_SPEED + (MAX_SPEED - CREEP_SPEED) * t

    #     # Beyond FAR → full speed
    #     return MAX_SPEED

    def _ramp_speed(
        self, current: float, target: float, dt: float
    ) -> float:
        """Apply acceleration / deceleration ramp."""
        if target > current:
            return min(current + ACCEL_RAMP * dt, target)
        elif target < current:
            return max(current - DECEL_RAMP * dt, target)
        return current

    # @staticmethod
    # def _phase_name(dist: float) -> str:
    #     if dist <= DIST_DEAD:  return "DEAD"
    #     if dist <= DIST_CLOSE: return "CLOSE"
    #     if dist <= DIST_FAR:   return "MID"
    #     return "FAR"

# ─────────────────────────────────────────────
#  HUD overlay helper  (optional, plug into main.py)
# ─────────────────────────────────────────────

def draw_guidance_hud(display, cmd: GuidanceCommand, x: int = 20, y: int = 60):
    import cv2

    PHASE_COLOR = {
        "FAR":   (0, 200, 255),   # orange
        "MID":   (0, 255, 200),   # cyan-green
        "CLOSE": (0, 255, 0),     # green
        "DEAD":  (0, 128, 0),     # dark green
        "LOST":  (0, 0, 255),     # red
        "IDLE":  (180, 180, 180), # grey
    }
    color = PHASE_COLOR.get(cmd.phase, (255, 255, 255))

    lines = [
        f"PHASE    : {cmd.phase}",
        f"DIST     : {cmd.est_dist:5.1f} m",
        f"SPEED    : {cmd.speed:5.2f} m/s",
        f"Vx/Vy/Vz : {cmd.vx:+.2f} / {cmd.vy:+.2f} / {cmd.vz:+.2f}",
        f"YAW RATE : {cmd.yaw_rate:+.1f} deg/s  (P:{cmd.yaw_p:+.1f} D:{cmd.yaw_d:+.1f})",
        f"TGT VEL  : {cmd.target_vel_x:+.0f} / {cmd.target_vel_y:+.0f} px/s",
        f"ERR x/y  : {cmd.err_x:+.0f} / {cmd.err_y:+.0f} px",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            display, line,
            (x, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50, color, 1, cv2.LINE_AA,
        )

    line_y = y + len(lines) * 22

    # ── Speed bar ──────────────────────────────────────────────────────
    bar_x, bar_y = x, line_y + 6
    bar_max = 150
    bar_fill = int(bar_max * min(cmd.speed / MAX_SPEED, 1.0))
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_max, bar_y + 8), (60, 60, 60), -1)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 8), color, -1)
    cv2.putText(display, "SPD", (bar_x + bar_max + 6, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ── Target direction arrow ─────────────────────────────────────────
    # Shows where the target is currently moving in the frame
    arrow_cx = bar_x + bar_max + 60
    arrow_cy = bar_y + 4
    arrow_len = 30
    spd_magnitude = math.hypot(cmd.target_vel_x, cmd.target_vel_y)

    if spd_magnitude > 5:   # only draw if target is actually moving
        norm = spd_magnitude
        dx = int(arrow_len * cmd.target_vel_x / norm)
        dy = int(arrow_len * cmd.target_vel_y / norm)
        cv2.arrowedLine(
            display,
            (arrow_cx, arrow_cy),
            (arrow_cx + dx, arrow_cy + dy),
            (0, 255, 255), 2, tipLength=0.4,
        )
        cv2.putText(display, "TGT", (arrow_cx - 10, arrow_cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # ── Yaw indicator bar (centred, ± MAX_YAW_RATE) ───────────────────
    yaw_bar_w  = 100
    yaw_bar_h  = 8
    yaw_bar_x  = bar_x
    yaw_bar_y  = bar_y + 18
    centre_x   = yaw_bar_x + yaw_bar_w // 2

    cv2.rectangle(display,
                  (yaw_bar_x, yaw_bar_y),
                  (yaw_bar_x + yaw_bar_w, yaw_bar_y + yaw_bar_h),
                  (60, 60, 60), -1)

    fill = int((yaw_bar_w / 2) * (cmd.yaw_rate / MAX_YAW_RATE))
    fill = max(-yaw_bar_w // 2, min(yaw_bar_w // 2, fill))

    if fill >= 0:
        cv2.rectangle(display, (centre_x, yaw_bar_y),
                      (centre_x + fill, yaw_bar_y + yaw_bar_h), (255, 100, 0), -1)
    else:
        cv2.rectangle(display, (centre_x + fill, yaw_bar_y),
                      (centre_x, yaw_bar_y + yaw_bar_h), (255, 100, 0), -1)

    # centre tick
    cv2.line(display, (centre_x, yaw_bar_y), (centre_x, yaw_bar_y + yaw_bar_h),
             (200, 200, 200), 1)
    cv2.putText(display, "YAW", (yaw_bar_x + yaw_bar_w + 6, yaw_bar_y + yaw_bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 100, 0), 1)