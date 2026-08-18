import cv2
import time
import socket
import json
import threading
import numpy as np
from tracker import LogicTracker
from flight_ctr import DroneGuidance
from px4_follow import PX4Follower
import math


# ============================================================
# GZ-TRANSPORT
# ============================================================

GZ_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

try:
    from gz.transport13 import Node
    import gz.msgs10.image_pb2 as image_pb2
    print("✅ gz-transport13 + gz-msgs10 OK")
except ImportError:
    print("❌ Không import được gz.transport13 / gz.msgs10")
    print("   Thử: pip install gz-transport13 gz-msgs10")
    import sys; sys.exit(1)


class GazeboCamera:
    """Nhận frame từ Gazebo qua gz-transport, thread-safe."""

    def __init__(self, topic: str):
        self._lock  = threading.Lock()
        self._frame = None
        self._node  = Node()

        ok = self._node.subscribe(image_pb2.Image, topic, self._callback)
        if not ok:
            print(f"❌ Không subscribe được topic: {topic}")
            import sys; sys.exit(1)
        print(f"📷 Subscribe: {topic}")

    def _callback(self, msg):
        w   = msg.width
        h   = msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.pixel_format_type == 1:       # L_INT8 grayscale
            img = raw.reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif msg.pixel_format_type == 3:     # RGB_INT8
            img = raw.reshape((h, w, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.pixel_format_type == 4:     # RGBA_INT8
            img = raw.reshape((h, w, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            try:
                img = raw.reshape((h, w, 3))
            except Exception:
                return

        # Only resize in callback if needed — avoids main-thread resize
        if w != 640 or h != 360:
            img = cv2.resize(img, (640, 360))

        with self._lock:
            self._frame = img  # no extra copy — we own this array

    def read(self):
        """Returns (True, frame) or (False, None). One copy only."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()


# ============================================================
# UDP — joystick state
# ============================================================

LISTEN_IP   = "0.0.0.0"
LISTEN_PORT = 5005

class JoystickState:
    def __init__(self):
        self._lock = threading.Lock()
        self.sb = "normal"

    def update(self, data: dict):
        with self._lock:
            self.sb = data.get("sb", "normal")

    def get_sb(self) -> str:
        with self._lock:
            return self.sb

def udp_listener(state: JoystickState):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    print(f"📡 Lắng nghe UDP {LISTEN_IP}:{LISTEN_PORT}...")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            payload = json.loads(data.decode())
            state.update(payload)
        except socket.timeout:
            continue
        except Exception:
            continue


# ============================================================
# ASYNC TRACKER WRAPPER
# Runs tracker.update() in a background thread.
# Main loop always gets the latest result without stalling.
# ============================================================

class AsyncTracker:

    def __init__(self):
        self._tracker     = LogicTracker()
        self._lock        = threading.Lock()

        # Latest tracker output
        self._result      = (False, None)

        # Frame queue: only keep newest (size=1, drop stale)
        self._pending     = None
        self._pending_lock = threading.Lock()
        self._has_pending  = threading.Event()

        self._running     = False
        self._thread      = None

    def init(self, frame: np.ndarray, bbox: tuple):
        """Initialise tracker. Call on lock transition."""
        # Stop any running worker first
        self._stop_worker()

        self._tracker.init(frame, bbox)
        self._result = (True, bbox)

        # Start worker
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker, daemon=True
        )
        self._thread.start()

    def request(self, frame: np.ndarray):
        """Submit a new frame for tracking. Drops stale frames automatically."""
        with self._pending_lock:
            self._pending = frame.copy()
        self._has_pending.set()

    def result(self):
        """Returns latest (success, box) — never blocks."""
        with self._lock:
            return self._result

    def stop(self):
        self._stop_worker()
        with self._lock:
            self._result = (False, None)

    def _stop_worker(self):
        if self._running:
            self._running = False
            self._has_pending.set()   # unblock worker if waiting
            if self._thread:
                self._thread.join(timeout=1.0)
        self._has_pending.clear()

    def _worker(self):
        while self._running:
            with self._pending_lock:
                frame = self._pending
                self._pending = None

            if frame is None:
                time.sleep(0.001)
                continue

            success, box = self._tracker.update(frame)

            with self._lock:
                self._result = (success, box)


# ============================================================
# MAIN
# ============================================================

gz_cam = GazeboCamera(GZ_TOPIC)

js_state = JoystickState()
t = threading.Thread(target=udp_listener, args=(js_state,), daemon=True)
t.start()

async_tracker = AsyncTracker()

guidance = DroneGuidance(frame_w=640, frame_h=360)

px4 = PX4Follower(ip="127.0.0.1", port=14580)

# Start pre-streaming setpoints immediately so PX4 is ready
px4.start_prestream(duration=2.0)

tracking  = False
box_size  = 30
bbox      = None
sb_prev   = "normal"
prev_time = 0
fps       = 0

cv2.namedWindow("Frame")
print("\n⏳ Chờ frame từ Gazebo...")

while True:

    ret, frame = gz_cam.read()

    if not ret:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for Gazebo camera...",
                    (120, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow("Frame", blank)
        if cv2.waitKey(30) == 27:
            break
        continue

    # frame is already 640×360 (resize moved to GazeboCamera._callback)
    h_frame, w_frame = frame.shape[:2]
    display = frame.copy()   # one copy for drawing — kept

    # ── Joystick state ────────────────────────────────────────
    sb = js_state.get_sb()

    # ── State transitions ─────────────────────────────────────
    if sb_prev != "tracking" and sb == "tracking":
        cx   = w_frame // 2
        cy   = h_frame // 2
        bbox = (cx - box_size // 2, cy - box_size // 2, box_size, box_size)

        async_tracker.init(frame, bbox)   # no extra .copy() — init() does it
        tracking = True
        px4.enter_offboard()              # non-blocking (background thread)
        print("🟢 LOCK → Tracking started")

    elif sb_prev == "tracking" and sb == "ready":
        tracking = False
        bbox     = None
        async_tracker.stop()
        px4.stop()
        px4.exit_offboard()
        print("🟡 UNLOCK → Ready")

    elif sb == "normal" and sb_prev != "normal":
        tracking = False
        bbox     = None
        async_tracker.stop()
        print("⚫ NORMAL mode")

    sb_prev = sb

    # ── READY: yellow center box ──────────────────────────────
    if sb == "ready":
        x1 = w_frame // 2 - box_size // 2
        y1 = h_frame // 2 - box_size // 2
        cv2.rectangle(display, (x1, y1),
                      (x1 + box_size, y1 + box_size), (0, 255, 255), 1)
        cv2.putText(display, "READY TO LOCK",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ── TRACKING ──────────────────────────────────────────────
    zoom_frame = None

    if tracking:
        # Submit frame to background tracker
        async_tracker.request(frame)

        # Read latest result — never blocks
        success, box = async_tracker.result()

        if success and box is not None:
            x, y, w, h = map(int, box)

            # Send velocity if offboard is ready
            if px4.offboard_enabled:
                cmd = guidance.compute(box, frame_w=640, frame_h=360)
                px4.send_velocity(
                    vx=cmd.vx,
                    vy=0.0,
                    vz=0.0,
                    yaw_rate=math.radians(cmd.yaw_rate)
                )

            if x < 0 or y < 0 or x + w > w_frame or y + h > h_frame:
                tracking = False
                cv2.putText(display, "LOST TARGET", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Seeker ROI — single resize directly to final size
                padding = 2.5
                cx_t = x + w // 2
                cy_t = y + h // 2
                zw   = int(w * padding)
                zh   = int(h * padding)
                x1   = max(cx_t - zw // 2, 0)
                y1   = max(cy_t - zh // 2, 0)
                x2   = min(cx_t + zw // 2, w_frame)
                y2   = min(cy_t + zh // 2, h_frame)

                zoom_roi = frame[y1:y2, x1:x2]
                if zoom_roi.shape[0] > 0 and zoom_roi.shape[1] > 0:
                    zoom_frame = cv2.resize(zoom_roi, (150, 150))  # one resize

        else:
            tracking = False

    # ── Seeker view overlay ───────────────────────────────────
    if zoom_frame is not None:
        h_d, w_d = display.shape[:2]
        seeker_w = seeker_h = 150
        x_off = w_d - seeker_w - 20
        y_off = h_d - seeker_h - 20
        display[y_off:y_off+seeker_h, x_off:x_off+seeker_w] = zoom_frame
        cv2.rectangle(display,
                      (x_off, y_off),
                      (x_off+seeker_w, y_off+seeker_h),
                      (0, 255, 0), 2)
        cv2.putText(display, "SEEKER VIEW",
                    (x_off, y_off - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── HUD ───────────────────────────────────────────────────
    cv2.putText(display, f"FPS: {int(fps)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    sb_label = {"normal": "NORMAL", "ready": "READY", "tracking": "TRACKING"}
    sb_color = {"normal": (128, 128, 128), "ready": (0, 255, 255), "tracking": (0, 255, 0)}
    cv2.putText(display,
                f"MODE: {sb_label.get(sb, sb.upper())}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                sb_color.get(sb, (255, 255, 255)), 1)

    cv2.imshow("Frame", display)

    # FPS — only on valid frames
    current_time = time.time()
    if prev_time != 0:
        delta = current_time - prev_time
        if delta > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / delta)
    prev_time = current_time

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()