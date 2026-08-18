"""
joystick_control.py
VUAV Joystick Control - RadioMaster GX12 → PX4 SITL
====================================================
Script chạy trên Windows để đọc joystick RadioMaster GX12
và gửi lệnh MANUAL_CONTROL tới PX4 SITL trong WSL2.

Đồng thời gửi trạng thái switch SB (A5) tới tracker_ui.py qua UDP.

QUAN TRỌNG: PX4 cần set COM_RC_IN_MODE = 1 (MavLinkOnly)
    pxh> param set COM_RC_IN_MODE 1
    pxh> param save

Cách dùng:
    pip install pygame pymavlink
    python joystick_control.py --target <WSL_IP>:14580
    python joystick_control.py --target <WSL_IP>:14580 --tracker-ip 127.0.0.1 --tracker-port 5005
"""

from __future__ import annotations
import argparse
import sys
import time
import os
import socket
import json
from typing import Any, Optional

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    import pygame
except ImportError:
    print("❌ Cần cài pygame: pip install pygame")
    sys.exit(1)

try:
    from pymavlink import mavutil
except ImportError:
    print("❌ Cần cài pymavlink: pip install pymavlink")
    sys.exit(1)


# ============================================================
# CẤU HÌNH JOYSTICK
# ============================================================

AXIS_ROLL      = 0
AXIS_PITCH     = 1
AXIS_THROTTLE  = 2
AXIS_YAW       = 3

AXIS_SA        = 4
AXIS_SB        = 5   # ← switch điều khiển tracking
AXIS_SD        = 6
AXIS_SC        = 7

INVERT_PITCH    = True
INVERT_THROTTLE = False
INVERT_ROLL     = False
INVERT_YAW      = False

BUTTON_ARM       = 0
BUTTON_DISARM    = 1
BUTTON_MODE_STAB = 4
BUTTON_MODE_POS  = 5

DEADZONE  = 0.05
SEND_RATE = 50   # Hz

# Ngưỡng nhận diện vị trí switch (3-position)
SWITCH_NEG = -0.5   # SB = -1.00  → Normal
SWITCH_MID =  0.3   # SB =  0.00  → Show box
SWITCH_POS =  0.7   # SB = +1.00  → Lock & Track

# ============================================================
# UDP TRACKER
# ============================================================

DEFAULT_TRACKER_IP   = "172.27.101.234"
DEFAULT_TRACKER_PORT = 5005


def read_switch(value: float) -> str:
    """Chuyển giá trị axis thành trạng thái switch: 'normal' | 'ready' | 'tracking'"""
    if value < SWITCH_NEG:
        return "normal"
    elif value < SWITCH_MID:
        return "ready"
    else:
        return "tracking"


class TrackerBridge:
    """Gửi trạng thái joystick tới tracker_ui.py qua UDP."""

    def __init__(self, ip: str, port: int):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sb_state = ""
        print(f"📡 TrackerBridge → {ip}:{port}")

    def send(self, sb_state: str, roll: int, pitch: int, throttle: int, yaw: int):
        payload = json.dumps({
            "sb":       sb_state,
            "roll":     roll,
            "pitch":    pitch,
            "throttle": throttle,
            "yaw":      yaw,
        })
        try:
            self.sock.sendto(payload.encode(), self.addr)
        except Exception:
            pass   # non-blocking, không crash nếu tracker chưa chạy

        if sb_state != self.last_sb_state:
            icons = {"normal": "⚫", "ready": "🟡", "tracking": "🟢"}
            print(f"\n  SB → {icons.get(sb_state,'?')} {sb_state.upper()}")
            self.last_sb_state = sb_state

    def close(self):
        self.sock.close()


# ============================================================


class JoystickController:
    def __init__(self, target: str, joystick_id: int = 0,
                 tracker_ip: str = DEFAULT_TRACKER_IP,
                 tracker_port: int = DEFAULT_TRACKER_PORT):
        self.target       = target
        self.joystick_id  = joystick_id
        self.connection: Optional[Any] = None
        self.joystick:   Optional[Any] = None
        self.running: bool = False
        self.armed:   bool = False

        self.roll:     int = 0
        self.pitch:    int = 0
        self.throttle: int = 0
        self.yaw:      int = 0
        self.sb_state: str = "normal"

        self.prev_buttons: dict[int, bool] = {}

        self.bridge = TrackerBridge(tracker_ip, tracker_port)

    # ----------------------------------------------------------
    # MAVLink
    # ----------------------------------------------------------

    def connect_mavlink(self) -> bool:
        print(f"🔗 Kết nối MAVLink tới udpout:{self.target}...")
        try:
            self.connection = mavutil.mavlink_connection(
                f"udpout:{self.target}",
                source_system=255,
                source_component=190,
            )
            print(f"✅ Kết nối tới {self.target}")
            self.send_heartbeat()
            return True
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False

    def send_heartbeat(self):
        if self.connection is None:
            return
        self.connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )

    # ----------------------------------------------------------
    # Joystick
    # ----------------------------------------------------------

    def init_joystick(self) -> bool:
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            print("❌ Không tìm thấy joystick!")
            return False

        print(f"\n🎮 {count} joystick(s):")
        for i in range(count):
            js = pygame.joystick.Joystick(i)
            js.init()
            print(f"   [{i}] {js.get_name()} - {js.get_numaxes()} axes, {js.get_numbuttons()} buttons")
            js.quit()

        self.joystick = pygame.joystick.Joystick(self.joystick_id)
        self.joystick.init()
        print(f"✅ Dùng: {self.joystick.get_name()}")
        return True

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < DEADZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)

    def read_joystick(self):
        if self.joystick is None:
            return
        pygame.event.pump()
        num_axes = self.joystick.get_numaxes()

        def axis(i):
            return self.joystick.get_axis(i) if i < num_axes else 0.0

        raw_roll     = axis(AXIS_ROLL)
        raw_pitch    = axis(AXIS_PITCH)
        raw_throttle = axis(AXIS_THROTTLE)
        raw_yaw      = axis(AXIS_YAW)
        raw_sb       = axis(AXIS_SB)

        if INVERT_ROLL:     raw_roll     = -raw_roll
        if INVERT_PITCH:    raw_pitch    = -raw_pitch
        if INVERT_THROTTLE: raw_throttle = -raw_throttle
        if INVERT_YAW:      raw_yaw      = -raw_yaw

        raw_roll  = self.apply_deadzone(raw_roll)
        raw_pitch = self.apply_deadzone(raw_pitch)
        raw_yaw   = self.apply_deadzone(raw_yaw)

        self.roll     = int(max(-1000, min(1000, raw_roll  * 1000)))
        self.pitch    = int(max(-1000, min(1000, raw_pitch * 1000)))
        self.yaw      = int(max(-1000, min(1000, raw_yaw   * 1000)))
        self.throttle = int(max(0, min(1000, (raw_throttle + 1.0) / 2.0 * 1000)))

        # SB switch state
        self.sb_state = read_switch(raw_sb)

    def _button_pressed(self, button_id: int) -> bool:
        if self.joystick is None:
            return False
        if button_id >= self.joystick.get_numbuttons():
            return False
        current  = bool(self.joystick.get_button(button_id))
        previous = self.prev_buttons.get(button_id, False)
        self.prev_buttons[button_id] = current
        return current and not previous

    def check_buttons(self):
        if self.joystick is None:
            return
        if self._button_pressed(BUTTON_ARM):
            self.arm()
        if self._button_pressed(BUTTON_DISARM):
            self.disarm()
        if self._button_pressed(BUTTON_MODE_STAB):
            self.set_mode("STABILIZED")
        if self._button_pressed(BUTTON_MODE_POS):
            self.set_mode("POSCTL")

    # ----------------------------------------------------------
    # MAVLink commands
    # ----------------------------------------------------------

    def arm(self):
        if self.connection is None:
            return
        print("\n🟢 ARM...")
        self.connection.mav.command_long_send(
            1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)
        self.armed = True

    def disarm(self):
        if self.connection is None:
            return
        print("\n🔴 DISARM...")
        self.connection.mav.command_long_send(
            1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0)
        self.armed = False

    def set_mode(self, mode_name: str):
        if self.connection is None:
            return
        modes = {"STABILIZED": 7, "POSCTL": 3, "MANUAL": 1, "ALTCTL": 2}
        custom_mode = modes.get(mode_name)
        if custom_mode is None:
            return
        print(f"\n🔄 Mode → {mode_name}")
        self.connection.mav.command_long_send(
            1, 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            custom_mode, 0, 0, 0, 0, 0)

    def send_manual_control(self):
        if self.connection is None:
            return
        self.connection.mav.manual_control_send(
            1,
            self.pitch,
            self.roll,
            self.throttle,
            self.yaw,
            0
        )

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------

    def print_status(self):
        sb_icons = {"normal": "⚫", "ready": "🟡", "tracking": "🟢"}
        arm_str  = "ARMED 🟢" if self.armed else "DISARMED 🔴"
        sb_str   = f"SB:{sb_icons.get(self.sb_state,'?')}"
        print(
            f"\r  R:{self.roll:+5d} P:{self.pitch:+5d} "
            f"T:{self.throttle:4d} Y:{self.yaw:+5d} | {sb_str} | {arm_str}  ",
            end="", flush=True
        )

    def run(self):
        self.running = True
        interval  = 1.0 / SEND_RATE
        last_print = 0.0
        last_hb    = 0.0

        print("\n" + "=" * 60)
        print("🎮 JOYSTICK → PX4 (MANUAL_CONTROL) + TRACKER BRIDGE")
        print("=" * 60)
        print(f"  Roll: A{AXIS_ROLL} | Pitch: A{AXIS_PITCH}")
        print(f"  Thr:  A{AXIS_THROTTLE} | Yaw:   A{AXIS_YAW}")
        print(f"  SB (tracking switch): A{AXIS_SB}")
        print(f"    -1.00 = Normal | 0.00 = Ready | +1.00 = Track")
        print(f"  ARM: B{BUTTON_ARM} | DISARM: B{BUTTON_DISARM}")
        print(f"  STAB: B{BUTTON_MODE_STAB} | POS: B{BUTTON_MODE_POS}")
        print("=" * 60)
        print("  ⚠️  PX4: param set COM_RC_IN_MODE 1")
        print("  Ctrl+C để thoát\n")

        try:
            while self.running:
                start = time.time()

                if start - last_hb >= 1.0:
                    self.send_heartbeat()
                    last_hb = start

                self.read_joystick()
                self.check_buttons()
                self.send_manual_control()

                # Gửi trạng thái tới tracker
                self.bridge.send(
                    self.sb_state,
                    self.roll, self.pitch,
                    self.throttle, self.yaw
                )

                now = time.time()
                if now - last_print >= 0.1:
                    self.print_status()
                    last_print = now

                elapsed = time.time() - start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        except KeyboardInterrupt:
            print("\n\n🛑 Dừng...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        if self.joystick is not None:
            self.joystick.quit()
        pygame.quit()
        self.bridge.close()
        if self.connection is not None:
            self.connection.close()
        print("✅ Đã dừng")


# ============================================================
# CALIBRATE
# ============================================================

def calibrate_mode(joystick_id: int = 0):
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("❌ Không tìm thấy joystick!")
        return
    js = pygame.joystick.Joystick(joystick_id)
    js.init()
    print(f"\n🔧 CALIBRATE - {js.get_name()}")
    print(f"   {js.get_numaxes()} axes, {js.get_numbuttons()} buttons\n")
    try:
        while True:
            pygame.event.pump()
            axes    = " | ".join(f"A{i}:{js.get_axis(i):+.3f}" for i in range(js.get_numaxes()))
            pressed = [str(i) for i in range(js.get_numbuttons()) if js.get_button(i)]
            btn     = f"Btn:[{','.join(pressed)}]" if pressed else "Btn:[]"
            sb_raw  = js.get_axis(AXIS_SB) if AXIS_SB < js.get_numaxes() else 0.0
            sb_st   = read_switch(sb_raw)
            print(f"\r  {axes} | {btn} | SB→{sb_st}    ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n✅ Done")
    finally:
        js.quit()
        pygame.quit()


# ============================================================
# ENTRY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VUAV Joystick → PX4 SITL + Tracker")
    parser.add_argument("--target",       "-t", default="127.0.0.1:14580",
                        help="PX4 MAVLink address (default: 127.0.0.1:14580)")
    parser.add_argument("--joystick",     "-j", type=int, default=0)
    parser.add_argument("--calibrate",    "-c", action="store_true")
    parser.add_argument("--tracker-ip",         default=DEFAULT_TRACKER_IP,
                        help="IP của máy chạy tracker_ui.py (default: 172.27.101.234)")
    parser.add_argument("--tracker-port", type=int, default=DEFAULT_TRACKER_PORT,
                        help="UDP port của tracker_ui.py (default: 5005)")
    args = parser.parse_args()

    print("=" * 60)
    print("  🎮 VUAV Joystick Control v3")
    print("=" * 60)

    if args.calibrate:
        calibrate_mode(args.joystick)
        return

    ctrl = JoystickController(
        target=args.target,
        joystick_id=args.joystick,
        tracker_ip=args.tracker_ip,
        tracker_port=args.tracker_port,
    )
    if not ctrl.init_joystick():
        sys.exit(1)
    if not ctrl.connect_mavlink():
        sys.exit(1)
    ctrl.run()


if __name__ == "__main__":
    main()