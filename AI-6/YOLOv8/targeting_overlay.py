"""
targeting_overlay.py - Targeting / guidance overlay for drone hobby project

Draws:
  - A target strike zone (purple rectangle) in the center of the frame
  - A line from the detected object center to the zone center
  - Direction offset in pixels (Left/Right, Up/Down)
  - Inside / Outside status
  - Lock-on state when target is close enough to frame center
  - Press ENTER to release lock
  - "Hedef Vurus Alani" label at the bottom of the zone
"""

import cv2
import numpy as np

# =====================================================
# ZONE CONFIG
# =====================================================
ZONE_COLOR           = (255, 0, 200)   # Purple (BGR)
ZONE_THICKNESS       = 1
ZONE_WIDTH_RATIO     = 0.55
ZONE_HEIGHT_RATIO    = 0.70

LINE_COLOR           = (0, 255, 0)     # Green
LINE_THICKNESS       = 1
LINE_DOT_RADIUS      = 5

LABEL_FONT           = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE          = 0.65
LABEL_THICKNESS      = 1
LABEL_COLOR          = (0, 0, 255)     # Red

ZONE_LABEL           = "Target Zone" 

# =====================================================
# LOCK-ON CONFIG
# =====================================================
LOCK_RADIUS_PX       = 40    # Pixel radius from frame center to trigger lock
LOCK_COLOR           = (0, 0, 255)
LOCK_CROSSHAIR_SIZE  = 30
LOCK_CROSSHAIR_THICK = 1
LOCK_TEXT            = "** LOCKED **"
LOCK_RELEASE_TEXT    = "Press ENTER to release"
LOCK_TEXT_SCALE      = 1.0
LOCK_TEXT_THICK      = 1

# =====================================================
# LOCK STATE
# =====================================================
_lock_counter = 0
_is_locked    = False
_in_range     = False   # True when target is close enough to lock
_locked_tid   = None    # ByteTrack ID of the locked target
_pending_tid  = None    # ByteTrack ID of target currently in range


def confirm_lock():
    """Call when X is pressed — locks if target is in range."""
    global _is_locked, _locked_tid
    print(f"X pressed — in_range={_in_range}, is_locked={_is_locked}")
    if _in_range and not _is_locked:
        _is_locked  = True
        _locked_tid = _pending_tid   # save the ByteTrack ID at time of lock
        print(f"Target locked (ID:{_locked_tid})")
    elif _is_locked:
        print("Already locked")
    else:
        print("Target not in range yet")


def get_locked_tid():
    """Return the track ID of the locked target, or None."""
    return _locked_tid


def is_locked() -> bool:
    return _is_locked


def release_lock():
    global _lock_counter, _is_locked, _in_range, _pending_tid, _locked_tid
    _lock_counter = 0
    _is_locked    = False
    _in_range     = False
    _locked_tid   = None
    print("Lock released")


def reset_lock():
    release_lock()


# =====================================================
# HELPERS
# =====================================================
def _zone_rect(frame_w, frame_h):
    zw = int(frame_w * ZONE_WIDTH_RATIO)
    zh = int(frame_h * ZONE_HEIGHT_RATIO)
    cx, cy = frame_w // 2, frame_h // 2
    x1 = cx - zw // 2
    y1 = cy - zh // 2
    return x1, y1, x1 + zw, y1 + zh


def _box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _draw_crosshair(frame, cx, cy, size, color, thickness):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)


def _draw_corner_brackets(frame, x1, y1, x2, y2, color, thickness, arm=20):
    pts = [
        ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),
        ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
        ((x1, y2 - arm), (x1, y2), (x1 + arm, y2)),
        ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),
    ]
    for p1, corner, p2 in pts:
        cv2.line(frame, p1, corner, color, thickness)
        cv2.line(frame, corner, p2, color, thickness)


def _draw_zone(frame, locked=False):
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _zone_rect(fw, fh)
    color = LOCK_COLOR if locked else ZONE_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, ZONE_THICKNESS)
    _draw_zone_label(frame, x1, y1, x2, y2, color)


def _draw_zone_label(frame, x1, y1, x2, y2, color):
    (tw, th), _ = cv2.getTextSize(ZONE_LABEL, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    lx = x1 + (x2 - x1 - tw) // 2
    cv2.putText(frame, ZONE_LABEL, (lx, y2 - 10),
                LABEL_FONT, LABEL_SCALE, color, LABEL_THICKNESS)


def _draw_press_x_prompt(frame, cx, cy):
    """Show a prompt when target is in lock range."""
    radius = LOCK_RADIUS_PX + 8
    cv2.circle(frame, (cx, cy), radius, (0, 200, 255), 2)
    text = "Press X to lock"
    (tw, th), _ = cv2.getTextSize(text, LABEL_FONT, 0.65, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + radius + 20),
                LABEL_FONT, 0.65, (0, 200, 255), 2)


# =====================================================
# MAIN DRAW FUNCTION
# =====================================================
def draw_targeting_overlay(frame, detections: list) -> None:
    global _lock_counter, _is_locked, _in_range

    fh, fw = frame.shape[:2]
    frame_cx, frame_cy = fw // 2, fh // 2
    zx1, zy1, zx2, zy2 = _zone_rect(fw, fh)
    zone_cx = (zx1 + zx2) // 2
    zone_cy = (zy1 + zy2) // 2

    if not detections:
        if not _is_locked:
            _lock_counter = 0
        _draw_zone(frame, locked=_is_locked)
        return

    primary        = detections[0]
    obj_cx, obj_cy = _box_center(primary['box'])

    if not _is_locked:
        dist      = np.hypot(obj_cx - frame_cx, obj_cy - frame_cy)
        _in_range   = dist <= LOCK_RADIUS_PX
        _pending_tid = primary.get('tid')   # remember who we might lock onto

    inside = zx1 <= obj_cx <= zx2 and zy1 <= obj_cy <= zy2

    dx = obj_cx - frame_cx
    dy = obj_cy - frame_cy

    # if _is_locked:
    #     h_label = f"Right:{abs(dx):.0f}px" if dx > 0 else f"Left:{abs(dx):.0f}px"
    #     v_label = f"Down:{abs(dy):.0f}px"  if dy > 0 else f"Up:{abs(dy):.0f}px"
    # else:
    #     h_label = f"Left:{abs(dx):.0f}px"  if dx > 0 else f"Right:{abs(dx):.0f}px"
    #     v_label = f"Up:{abs(dy):.0f}px"    if dy > 0 else f"Down:{abs(dy):.0f}px"

    if _is_locked:
        _draw_corner_brackets(frame, zx1, zy1, zx2, zy2, LOCK_COLOR, ZONE_THICKNESS + 1)
        _draw_crosshair(frame, frame_cx, frame_cy, LOCK_CROSSHAIR_SIZE, LOCK_COLOR, LOCK_CROSSHAIR_THICK)

        x1, y1, x2, y2 = primary['box']
        radius = int(max(x2 - x1, y2 - y1) // 2 * 1.2)
        cv2.circle(frame, (obj_cx, obj_cy), radius, LOCK_COLOR, 2)

        (tw, th), _ = cv2.getTextSize(LOCK_TEXT, LABEL_FONT, LOCK_TEXT_SCALE, LOCK_TEXT_THICK)
        cv2.putText(frame, LOCK_TEXT, (frame_cx - tw // 2, zy1 - 30),
                    LABEL_FONT, LOCK_TEXT_SCALE, LOCK_COLOR, LOCK_TEXT_THICK)

        (rw, rh), _ = cv2.getTextSize(LOCK_RELEASE_TEXT, LABEL_FONT, 0.55, 1)
        cv2.putText(frame, LOCK_RELEASE_TEXT, (frame_cx - rw // 2, zy1 - 8),
                    LABEL_FONT, 0.55, LOCK_COLOR, 1)

        _draw_zone_label(frame, zx1, zy1, zx2, zy2, LOCK_COLOR)

    else:
        _draw_zone(frame, locked=False)
        cv2.line(frame, (obj_cx, obj_cy), (zone_cx, zone_cy), LINE_COLOR, LINE_THICKNESS)
        cv2.circle(frame, (zone_cx, zone_cy), LINE_DOT_RADIUS, LINE_COLOR, -1)

        # status = "Inside" if inside else "Outside"
        # cv2.putText(frame, status, (zx1 + 10, zy1 + 40), LABEL_FONT, 1.2, LABEL_COLOR, 2)

        if _in_range:
            _draw_press_x_prompt(frame, frame_cx, frame_cy)

    color = LOCK_COLOR if _is_locked else LABEL_COLOR
    # cv2.putText(frame, h_label, (zx1 + 10, zy2 - 80),
    #             LABEL_FONT, LABEL_SCALE, color, LABEL_THICKNESS)
    # cv2.putText(frame, v_label, (zx1 + 10, zy2 - 55),
    #             LABEL_FONT, LABEL_SCALE, color, LABEL_THICKNESS)