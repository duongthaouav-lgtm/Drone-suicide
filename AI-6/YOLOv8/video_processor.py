"""
video_processor.py - 3-thread pipeline for maximum FPS on Pi 5

Thread 1 (camera)    : reads raw frames → frame_queue
Thread 2 (inference) : YOLO inference   → result_queue  
Thread 3 (display)   : shows results    → handles keys

This decouples camera I/O, inference, and display completely.
Each runs at its own speed — no thread waits for another.
"""
import os
import time
import threading
import queue
from collections import defaultdict

import cv2

import config as cfg
import perf_monitor
import frame_processor as fp
from targeting_overlay import release_lock, confirm_lock


# Queue sizes — small keeps latency low, large keeps throughput high
_FRAME_QUEUE_SIZE  = 2   # camera → inference (small = low latency)
_RESULT_QUEUE_SIZE = 2   # inference → display (small = low latency)


# =====================================================
# CAMERA OPEN HELPER
# =====================================================
def _open_camera(path) -> cv2.VideoCapture:
    """
    Open a camera device with Pi 5 optimisations.
    Works for /dev/video0, 0, or any V4L2 device path.
    """
    # Use V4L2 backend on Linux for lower latency
    if isinstance(path, str) and path.startswith('/dev/'):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    elif isinstance(path, int) or (isinstance(path, str) and path.isdigit()):
        cap = cv2.VideoCapture(int(path), cv2.CAP_V4L2)
    else:
        return None   # not a camera

    if not cap.isOpened():
        return None

    # ── Pi 5 camera optimisations ─────────────────────────────────────────────
    # Set resolution to match inference size — avoid unnecessary upscaling
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          cfg.CAMERA_FPS)

    # Use MJPG format — much faster than raw YUYV on USB cameras
    # MJPG compresses in hardware, reducing USB bandwidth significantly
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Minimize internal buffer — reduces latency (default is 3-4 frames)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    print(f"   Buffer size: 1 frame (low latency mode)")
    return cap


# =====================================================
# PUBLIC ENTRY POINT
# =====================================================
def process_video(model, device: str, logger=None):
    video_path = cfg.VIDEO_PATH

    # ── Camera input (integer index or /dev/videoX) ──────────────────────────
    is_camera = (
        video_path in ("0", 0)
        or (isinstance(video_path, str) and video_path.startswith('/dev/video'))
        or (isinstance(video_path, str) and video_path.isdigit())
    )
    if is_camera:
        print("📹 Opening camera...")
        cap = _open_camera(video_path)
        if cap is None or not cap.isOpened():
            print("❌ Cannot open camera")
            return
        _run_single_video(cap, "camera", model, device, logger)
        return

    if os.path.isdir(video_path):
        print(f"📁 Processing folder: {video_path}")
        exts = (".mp4", ".avi", ".mov", ".mkv")
        files = [f for f in os.listdir(video_path) if f.lower().endswith(exts)]
        if not files:
            print("❌ No video files found in folder")
            return
        for filename in files:
            full_path = os.path.join(video_path, filename)
            print(f"\n🚀 Processing: {filename}")
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"❌ Cannot open: {filename}")
                continue
            _run_single_video(cap, filename, model, device, logger)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return
    _run_single_video(cap, os.path.basename(video_path), model, device, logger)


# =====================================================
# INTERNAL: SINGLE VIDEO PIPELINE
# =====================================================
def _run_single_video(cap, video_name: str, model, device: str, logger=None):

    fps_src     = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total_frames > 0:
        print(f"📹 {video_name}: {width}x{height} @ {fps_src}fps, {total_frames} frames")
    else:
        print(f"📹 {video_name}: {width}x{height} @ {fps_src}fps (live)")

    if logger:
        logger.info("=" * 60)
        logger.info(f"Processing: {video_name}")
        logger.info(f"  Resolution: {width}x{height}, FPS: {fps_src}, Frames: {total_frames}")
        logger.info("=" * 60)

    fp.reset_state()
    perf_monitor.reset()

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if cfg.SAVE_RESULT:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(cfg.OUTPUT_DIR, f"output_{video_name}")
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        print(f"💾 Saving output to: {out_path}")

    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WINDOW_NAME, 1280, 720)
    print("\n🎮 Controls:  SPACE = pause/resume   ESC/Q = quit   X = lock   ENTER = release\n")

    # ── Shared state between threads ──────────────────────────────────────────
    frame_queue  = queue.Queue(maxsize=_FRAME_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=_RESULT_QUEUE_SIZE)

    stop_event   = threading.Event()   # signals all threads to stop
    pause_event  = threading.Event()   # signals inference to pause

    # Shared FPS counter (inference thread writes, display thread reads)
    fps_state = {'current': 0.0, 'frame_count': 0}
    fps_lock  = threading.Lock()

    # Session stats (inference thread accumulates)
    session_stats = {
        'frames_with_det': 0,
        'frames_no_det':   0,
        'class_stats':     defaultdict(int),
        'all_track_ids':   set(),
        'total_detections': 0,
    }
    stats_lock = threading.Lock()

    video_start = time.time()

    # =====================================================
    # THREAD 1 — CAMERA READER
    # =====================================================
    def camera_thread():
        frame_idx  = 0
        is_live    = (total_frames == 0)   # True for cameras, False for video files

        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.02)
                continue

            if is_live:
                # For live cameras: grab() advances the buffer without decoding,
                # then retrieve() decodes only the frame we actually want.
                # This drains the camera buffer so we always get the LATEST frame.
                grabbed = cap.grab()
                if not grabbed:
                    try:
                        frame_queue.put(None, timeout=1.0)
                    except queue.Full:
                        pass
                    break
                # Only decode if inference thread is ready for a new frame
                if not frame_queue.full():
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    frame_idx += 1
                    frame_queue.put((frame_idx, frame))  # non-blocking, queue not full
                # else: just grabbed and discarded — keeps buffer fresh
            else:
                # For video files: use normal read()
                ret, frame = cap.read()
                if not ret:
                    try:
                        frame_queue.put(None, timeout=1.0)
                    except queue.Full:
                        pass
                    break
                frame_idx += 1
                try:
                    frame_queue.put((frame_idx, frame), timeout=0.5)
                except queue.Full:
                    pass  # drop frame — inference can't keep up

    # =====================================================
    # THREAD 2 — INFERENCE
    # =====================================================
    def inference_thread():
        frame_times = []

        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                # End of video — forward sentinel to display
                try:
                    result_queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
                break

            frame_idx, frame = item

            t0 = time.perf_counter()
            annotated, stats = fp.process_frame(
                model, device, frame,
                frame_idx, total_frames,
                fps_state['current'],
                logger
            )
            elapsed = time.perf_counter() - t0

            # Update FPS
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_t = sum(frame_times) / len(frame_times)
            with fps_lock:
                fps_state['current']     = 1.0 / avg_t if avg_t > 0 else 0.0
                fps_state['frame_count'] = frame_idx

            # Accumulate session stats
            with stats_lock:
                if stats['detection_count'] > 0:
                    session_stats['frames_with_det'] += 1
                    session_stats['total_detections'] += stats['detection_count']
                    for cls, cnt in stats['class_counts'].items():
                        session_stats['class_stats'][cls] += cnt
                else:
                    session_stats['frames_no_det'] += 1
                session_stats['all_track_ids'].update(stats['track_ids'])

            try:
                result_queue.put((annotated, stats), timeout=0.5)
            except queue.Full:
                pass  # drop result if display is backed up

    # =====================================================
    # THREAD 3 — DISPLAY (runs on main thread for cv2 compatibility)
    # =====================================================
    def display_loop():
        last_frame = None

        while not stop_event.is_set():
            try:
                item = result_queue.get(timeout=0.1)
            except queue.Empty:
                # Show last frame while waiting for next result
                if last_frame is not None:
                    cv2.imshow(cfg.WINDOW_NAME, last_frame)
                key = cv2.waitKey(1) & 0xFF
                _handle_key(key, pause_event, stop_event)
                continue

            if item is None:
                # End of video
                break

            annotated, stats = item
            last_frame = annotated

            if writer:
                writer.write(annotated)

            cv2.imshow(cfg.WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            _handle_key(key, pause_event, stop_event)

    # ── Start threads ─────────────────────────────────────────────────────────
    t_camera    = threading.Thread(target=camera_thread,    daemon=True, name="camera")
    t_inference = threading.Thread(target=inference_thread, daemon=True, name="inference")

    t_camera.start()
    t_inference.start()

    # Display runs on main thread (cv2.imshow requires main thread on most systems)
    display_loop()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    stop_event.set()
    t_camera.join(timeout=2.0)
    t_inference.join(timeout=2.0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - video_start
    with fps_lock:
        frame_count = fps_state['frame_count']
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    perf    = perf_monitor.get_summary()

    print("\n" + "=" * 60)
    print("📊 SESSION STATISTICS")
    print("=" * 60)
    print(f"   Frames processed : {frame_count}")
    print(f"   Processing time  : {total_time:.2f}s")
    print(f"   Average FPS      : {avg_fps:.2f}")

    if perf:
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"   Avg inference : {perf['avg_inference_ms']:.2f} ms")
        print(f"   Avg total     : {perf['avg_total_ms']:.2f} ms")
        print(f"   Min inference : {perf['min_inference_ms']:.2f} ms")
        print(f"   Max inference : {perf['max_inference_ms']:.2f} ms")
        print(f"   Est. FPS      : {perf['fps']:.2f}")

    print("=" * 60)

    if logger:
        logger.info("FINAL STATISTICS")
        logger.info(f"  Frames: {frame_count}, Time: {total_time:.2f}s, FPS: {avg_fps:.2f}")
        with stats_lock:
            if cfg.LOG_DETECTION_STATS:
                logger.info(f"  Frames w/ detections: {session_stats['frames_with_det']}")
                logger.info(f"  Frames w/o detections: {session_stats['frames_no_det']}")
                for cls, cnt in sorted(session_stats['class_stats'].items(), key=lambda x: -x[1]):
                    logger.info(f"    {cls}: {cnt}")
            if cfg.LOG_TRACKING_STATS and cfg.ENABLE_TRACKING:
                logger.info(f"  Unique track IDs: {len(session_stats['all_track_ids'])}")
        logger.info("=" * 60)


# =====================================================
# KEY HANDLER
# =====================================================
def _handle_key(key: int, pause_event: threading.Event, stop_event: threading.Event):
    if key == 255:
        return
    if key in (27, ord('q')):
        print("👋 Exiting.")
        stop_event.set()
    elif key == ord(' '):
        if pause_event.is_set():
            pause_event.clear()
            print("▶️ Resumed")
        else:
            pause_event.set()
            print("⏸️ Paused")
    elif key == 13:
        release_lock()
    elif key in (ord('x'), ord('X')):
        confirm_lock()