# px4_follow.py

from pymavlink import mavutil
import time
import math
import threading


class PX4Follower:

    def __init__(self, ip="127.0.0.1", port=14580):

        print(f"[PX4] Connecting udpout:{ip}:{port}")

        self.master = mavutil.mavlink_connection(
            f"udpout:{ip}:{port}",
            source_system=250
        )

        self.target_system    = 1
        self.target_component = 1

        self.offboard_enabled = False

        # ── NEW: background state ──────────────────────────────
        self._offboard_ready  = threading.Event()
        self._prestream_active = False
        self._prestream_thread = None
        self._lock = threading.Lock()
        # ───────────────────────────────────────────────────────

        print("[PX4] Ready")

    # ----------------------------------------------------------
    # Pre-stream: call this as early as possible (e.g. on init)
    # Sends zero setpoints in background so PX4 accepts OFFBOARD
    # ----------------------------------------------------------

    def start_prestream(self, duration: float = 2.0, rate_hz: float = 20.0):
        """
        Start sending zero-velocity setpoints in background immediately.
        Call this right after PX4Follower() is created — not when the
        user triggers tracking. By the time tracking starts, PX4 has
        already seen enough setpoints.
        """
        if self._prestream_active:
            return

        self._prestream_active = True

        def _stream():
            interval = 1.0 / rate_hz
            end      = time.time() + duration
            while time.time() < end and self._prestream_active:
                self.send_velocity(0.0, 0.0, 0.0, 0.0)
                time.sleep(interval)

        self._prestream_thread = threading.Thread(
            target=_stream, daemon=True
        )
        self._prestream_thread.start()
        print("[PX4] Pre-stream started in background")

    # ----------------------------------------------------------
    # OFFBOARD — non-blocking, runs in background thread
    # ----------------------------------------------------------

    def enter_offboard(self):
        """Non-blocking. Returns immediately; sets _offboard_ready when done."""

        with self._lock:
            if self.offboard_enabled:
                return

        def _do_enter():
            print("[PX4] Entering OFFBOARD (background)...")

            # If pre-stream already ran, this loop is very short
            # Just ensure at least 10 setpoints have gone out
            for _ in range(10):
                self.send_velocity(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)   # 10 × 50ms = 0.5s max (vs old 1.0s)

            print("[PX4] Switching to OFFBOARD mode")

            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                6,   # OFFBOARD
                0, 0, 0, 0, 0
            )

            # ── Wait for ACK instead of sleeping blindly ──────
            ack_received = self._wait_for_ack(
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                timeout=2.0
            )

            if ack_received:
                print("[PX4] OFFBOARD confirmed via ACK")
            else:
                print("[PX4] ACK timeout — assuming OFFBOARD OK")

            with self._lock:
                self.offboard_enabled = True

            self._offboard_ready.set()   # signal main thread
            print("[PX4] OFFBOARD ready")

        t = threading.Thread(target=_do_enter, daemon=True)
        t.start()

    def _wait_for_ack(self, command: int, timeout: float = 2.0) -> bool:
        """
        Wait for COMMAND_ACK matching `command`.
        Returns True if ACK received, False on timeout.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            msg = self.master.recv_match(
                type="COMMAND_ACK",
                blocking=False
            )

            if msg and msg.command == command:
                return True

            time.sleep(0.02)

        return False

    # ----------------------------------------------------------
    # POSCTL
    # ----------------------------------------------------------

    def exit_offboard(self):
        try:
            print("[PX4] Returning POSCTL")
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                3,
                0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"[PX4] Mode switch failed: {e}")

        with self._lock:
            self.offboard_enabled = False

        self._offboard_ready.clear()

    # ----------------------------------------------------------
    # Velocity command (unchanged)
    # ----------------------------------------------------------

    def send_velocity(self, vx, vy, vz, yaw_rate):
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self.master.mav.set_position_target_local_ned_send(
            0,
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0,
            float(yaw_rate)
        )

    # ----------------------------------------------------------
    # Stop
    # ----------------------------------------------------------

    def stop(self):
        self.send_velocity(0.0, 0.0, 0.0, 0.0)

    # ----------------------------------------------------------
    # Helper: block until offboard is ready (optional use)
    # ----------------------------------------------------------

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        return self._offboard_ready.wait(timeout=timeout)