"""
benchmark.py - Detailed timing breakdown to find the real bottleneck
Run: python benchmark.py
"""
import time
import cv2
import numpy as np

import os
import config as cfg

# Apply hardware overrides FIRST — this sets USE_OV=True on Pi 5
import hardware
hardware.apply_hardware_overrides()

print("=" * 55)
print("BENCHMARK: Finding the real bottleneck on Pi 5")
print(f"  USE_OV        : {cfg.USE_OV}")
print(f"  OV_MODEL_PATH : {cfg.OV_MODEL_PATH}")
print("=" * 55)

from ultralytics import YOLO

# Load model
if cfg.USE_OV and os.path.exists(cfg.OV_MODEL_PATH):
    print(f"\n📦 Loading OpenVINO INT8: {cfg.OV_MODEL_PATH}")
    model = YOLO(cfg.OV_MODEL_PATH, task='detect')
else:
    print(f"\n📦 Loading PyTorch: {cfg.MODEL_PATH}")
    if not cfg.USE_OV:
        print("   ⚠️  USE_OV=False — set USE_OV=True in config.py to test OpenVINO")
    elif not os.path.exists(cfg.OV_MODEL_PATH):
        print(f"   ⚠️  Model not found at: {cfg.OV_MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH, task='detect')

dummy = np.zeros((640, 640, 3), dtype=np.uint8)

# Warmup
print("🔥 Warming up...")
for _ in range(5):
    model.predict(dummy, imgsz=640, verbose=False, device='cpu')

print("\n⏱️  Timing breakdown (average of 30 runs):\n")

# 1. Pure inference only
times = []
for _ in range(30):
    t = time.perf_counter()
    model.predict(dummy, imgsz=640, verbose=False, device='cpu',
                  conf=0.15, iou=0.45)
    times.append(time.perf_counter() - t)
inf_ms = np.mean(times) * 1000
print(f"  1. Pure inference (predict):     {inf_ms:.1f} ms  →  {1000/inf_ms:.1f} FPS")

# 2. Inference + unpack boxes
times = []
for _ in range(30):
    t = time.perf_counter()
    results = model.predict(dummy, imgsz=640, verbose=False, device='cpu',
                            conf=0.15, iou=0.45)
    _ = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else None
    times.append(time.perf_counter() - t)
unpack_ms = np.mean(times) * 1000
print(f"  2. Inference + unpack boxes:     {unpack_ms:.1f} ms  →  {1000/unpack_ms:.1f} FPS")

# 3. Frame resize overhead (640→original)
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
times = []
for _ in range(30):
    t = time.perf_counter()
    cv2.resize(frame, (640, 640))
    times.append(time.perf_counter() - t)
resize_ms = np.mean(times) * 1000
print(f"  3. Resize 1280x720 → 640x640:   {resize_ms:.1f} ms")

# 4. Frame copy (annotated = frame.copy())
times = []
for _ in range(30):
    t = time.perf_counter()
    _ = frame.copy()
    times.append(time.perf_counter() - t)
copy_ms = np.mean(times) * 1000
print(f"  4. Frame copy (1280x720):        {copy_ms:.1f} ms")

# 5. cv2.imshow equivalent (encode to JPEG)
times = []
for _ in range(30):
    t = time.perf_counter()
    cv2.imencode('.jpg', frame)
    times.append(time.perf_counter() - t)
show_ms = np.mean(times) * 1000
print(f"  5. Frame encode (display cost):  {show_ms:.1f} ms")

# 6. Full pipeline estimate
total_ms = inf_ms + resize_ms + copy_ms
print(f"\n  Estimated total per frame:       {total_ms:.1f} ms  →  {1000/total_ms:.1f} FPS")

# 7. CPU frequency
print("\n💻 System:")
try:
    with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq') as f:
        freq = int(f.read().strip()) // 1000
        print(f"  CPU frequency: {freq} MHz (max 2400 MHz)")
    with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor') as f:
        gov = f.read().strip()
        print(f"  CPU governor:  {gov}")
except:
    pass

try:
    import subprocess
    temp = subprocess.check_output(['vcgencmd', 'measure_temp']).decode().strip()
    print(f"  Temperature:   {temp}")
except:
    pass

try:
    import torch
    print(f"  Torch threads: {torch.get_num_threads()}")
except:
    pass

print("\n✅ Done")
print("=" * 55)