"""
test_guide.py — Visual simulator for flight_ctr.py 
====================================================
Runs WITHOUT a drone, WITHOUT a video file, WITHOUT a tracker.

A fake target bounces around the frame. You can:
  • Watch the guidance HUD respond in real-time
  • Switch between movement patterns (keys 1-4)
  • Click anywhere to teleport the target (tests sudden direction change)
  • Press SPACE to pause/resume
  • Press R to reset

What each indicator tells you:
  • Green box      = fake target
  • Cyan arrow     = direction target is currently moving
  • YAW bar        = drone yaw command (left ◄ centre ► right)
  • SPD bar        = forward speed (0 → MAX)
  • P / D values   = P reacts to position, D predicts the turn
"""

import cv2
import math
import time
import numpy as np
from flight_ctr import DroneGuidance, draw_guidance_hud, MAX_SPEED
from px4_follow import PX4Follower

# ─────────────────────────────────────────────
#  Simulation settings
# ─────────────────────────────────────────────
W, H = 640, 360
BOX_W, BOX_H = 30, 50          # fake target bounding box size
FPS_TARGET = 30

# ─────────────────────────────────────────────
#  Movement patterns
# ─────────────────────────────────────────────
PATTERNS = {
    "1": "Straight across (←)",
    "2": "S-curve (left then right)",
    "3": "Circle",
    "4": "Random jitter",
    "5": "Slow approach (gets bigger)",
}


class FakeTarget:
    """Generates a synthetic bounding box that moves around the frame."""

    def __init__(self):
        self.cx = W // 2
        self.cy = H // 2
        self.t  = 0.0
        self.pattern = "1"
        self.box_scale = 1.0      # simulates distance (bigger = closer)
        self._rand_vx = 0.0
        self._rand_vy = 0.0
        self._rand_timer = 0.0

    def update(self, dt: float):
        self.t += dt

        if self.pattern == "1":
            # Straight across left → right → left
            self.cx = W // 2 + int(math.sin(self.t * 0.6) * (W * 0.38))
            self.cy = H // 2

        elif self.pattern == "2":
            # S-curve
            self.cx = W // 2 + int(math.sin(self.t * 0.7) * (W * 0.35))
            self.cy = H // 2 + int(math.sin(self.t * 1.4) * (H * 0.25))

        elif self.pattern == "3":
            # Circle
            r = min(W, H) * 0.30
            self.cx = int(W // 2 + r * math.cos(self.t * 0.8))
            self.cy = int(H // 2 + r * math.sin(self.t * 0.8))

        elif self.pattern == "4":
            # Random direction changes
            self._rand_timer -= dt
            if self._rand_timer <= 0:
                angle = np.random.uniform(0, 2 * math.pi)
                spd   = np.random.uniform(60, 160)
                self._rand_vx = math.cos(angle) * spd
                self._rand_vy = math.sin(angle) * spd
                self._rand_timer = np.random.uniform(0.4, 1.2)

            self.cx = int(np.clip(self.cx + self._rand_vx * dt, BOX_W, W - BOX_W))
            self.cy = int(np.clip(self.cy + self._rand_vy * dt, BOX_H, H - BOX_H))

        elif self.pattern == "5":
            # Slow approach: target stays centre but grows (simulates closing distance)
            self.cx = W // 2 + int(math.sin(self.t * 0.3) * 30)
            self.cy = H // 2
            self.box_scale = 1.0 + (self.t % 10) * 0.18  # resets every 10s

        # Keep inside frame
        self.cx = int(np.clip(self.cx, BOX_W, W - BOX_W))
        self.cy = int(np.clip(self.cy, BOX_H, H - BOX_H))

    def get_bbox(self):
        bw = int(BOX_W * self.box_scale)
        bh = int(BOX_H * self.box_scale)
        x = self.cx - bw // 2
        y = self.cy - bh // 2
        return (x, y, bw, bh)

    def teleport(self, cx, cy):
        self.cx = cx
        self.cy = cy


# ─────────────────────────────────────────────
#  Main test loop
# ─────────────────────────────────────────────

def draw_crosshair(frame, cx, cy, size=12, color=(0, 255, 0), thickness=1):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)


def draw_frame_centre_cross(frame):
    """Show where 'locked on' would be — the drone's aim point."""
    cx, cy = W // 2, H // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (80, 80, 80), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (80, 80, 80), 1)
    cv2.circle(frame, (cx, cy), 4, (80, 80, 80), 1)


def draw_error_line(frame, cmd):
    """Draw line from frame centre to target centre — the error vector."""
    fc = (W // 2, H // 2)
    tc = (int(W // 2 + cmd.err_x), int(H // 2 + cmd.err_y))
    cv2.arrowedLine(frame, fc, tc, (0, 180, 255), 1, tipLength=0.2)


def draw_legend(frame, current_pattern, paused):
    lines = [
        "KEYS:",
        "1-5 : change pattern",
        "CLICK : teleport target",
        "SPACE : pause / resume",
        "R    : reset",
        "ESC  : quit",
        "",
        f"Pattern: {PATTERNS.get(current_pattern, '?')}",
        "PAUSED" if paused else "",
    ]
    for i, line in enumerate(lines):
        color = (0, 0, 255) if line == "PAUSED" else (160, 160, 160)
        cv2.putText(frame, line, (W - 210, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


click_pos = None

def on_mouse(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)


def main():
    global click_pos

    target   = FakeTarget()
    guidance = DroneGuidance(frame_w=W, frame_h=H)

    px4 = PX4Follower(
        ip="127.0.0.1",
        port=14580
    )
    
    cv2.namedWindow("Guidance Test")
    cv2.setMouseCallback("Guidance Test", on_mouse)

    prev_time = time.perf_counter()
    fps_disp  = 0.0
    paused    = False
    cmd       = None

    print("=" * 50)
    print("  Guidance Test Running")
    print("  Keys: 1-5 pattern | SPACE pause | R reset | ESC quit")
    print("  Click anywhere to teleport the target")
    print("=" * 50)

    while True:
        now = time.perf_counter()
        dt  = now - prev_time
        prev_time = now

        # smooth FPS display
        if dt > 0:
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt)

        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # subtle grid
        for gx in range(0, W, 80):
            cv2.line(frame, (gx, 0), (gx, H), (20, 20, 20), 1)
        for gy in range(0, H, 80):
            cv2.line(frame, (0, gy), (W, gy), (20, 20, 20), 1)

        # handle mouse teleport
        if click_pos is not None:
            target.teleport(*click_pos)
            click_pos = None

        if not paused:
            target.update(dt)

        bbox = target.get_bbox()
        cmd  = guidance.compute(bbox, dt=max(dt, 1e-4))

        px4.send_velocity(
            cmd.vx,
            cmd.vy,
            cmd.vz,
            math.radians(cmd.yaw_rate)
        )

        # ── Draw fake target box ────────────────────────────────────────
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        draw_crosshair(frame, target.cx, target.cy, size=8, color=(0, 220, 0))

        # ── Draw error line & frame cross ───────────────────────────────
        draw_frame_centre_cross(frame)
        draw_error_line(frame, cmd)

        # ── Guidance HUD ────────────────────────────────────────────────
        draw_guidance_hud(frame, cmd, x=14, y=58)

        # ── FPS ─────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {int(fps_disp)}",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

        # ── Console-style log strip at bottom ───────────────────────────
        log = (f"yaw={cmd.yaw_rate:+5.1f} deg/s  "
               f"spd={cmd.speed:.2f} m/s  "
               f"dist={cmd.est_dist:.1f} m  "
               f"vel=({cmd.target_vel_x:+.0f},{cmd.target_vel_y:+.0f}) px/s  "
               f"phase={cmd.phase}")
        cv2.rectangle(frame, (0, H - 22), (W, H), (15, 15, 15), -1)
        cv2.putText(frame, log, (8, H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        draw_legend(frame, target.pattern, paused)

        cv2.imshow("Guidance Test", frame)

        key = cv2.waitKey(max(1, int(1000 / FPS_TARGET) - int(dt * 1000)))

        if key == 27:                       # ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r') or key == ord('R'):
            target   = FakeTarget()
            guidance = DroneGuidance(frame_w=W, frame_h=H)
            print("Reset.")
        elif chr(key & 0xFF) in PATTERNS:
            target.pattern = chr(key & 0xFF)
            target.box_scale = 1.0
            print(f"Pattern → {PATTERNS[target.pattern]}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()