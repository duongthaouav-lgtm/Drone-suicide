"""
main.py - Entry point for YOLO Detection & Tracking System

Run:
    python main.py

Export to OpenVINO (do once on desktop, copy folder to Pi):
    yolo export model=best.pt format=openvino opset=12
    # Creates folder: best_openvino_model/
    # Install on Pi: pip install openvino
"""
import os
import numpy as np
import torch

# ── 1. Config ─────────────────────────────────────────────────────────────────
import config as cfg

# ── 2. Hardware detection & config overrides ──────────────────────────────────
import hardware
hardware.apply_hardware_overrides()
device = hardware.select_device()

# ── 3. Logger ─────────────────────────────────────────────────────────────────
import logger as log_module
log_module.logger = log_module.setup_logger()
logger = log_module.logger

# ── 4. Model loading ──────────────────────────────────────────────────────────
from ultralytics import YOLO
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Validate OpenVINO path before committing to it
if cfg.USE_OV:
    if not os.path.exists(cfg.OV_MODEL_PATH):
        print(f"⚠️  OpenVINO model not found at: {cfg.OV_MODEL_PATH}")
        print(f"   Please set OV_MODEL_PATH correctly in config.py")
        print(f"   Export with: yolo export model=best.pt format=openvino opset=12")
        print(f"   Falling back to PyTorch...")
        cfg.USE_OV = False
    else:
        print(f"\n📦 Loading OpenVINO model: {cfg.OV_MODEL_PATH}")

if cfg.USE_OV:
    model  = YOLO(cfg.OV_MODEL_PATH, task='detect')
    device = 'cpu'

    # ── Warm up: run one dummy inference so model.names loads ─────────────────
    print("🔥 Warming up OpenVINO model...")
    dummy = np.zeros((cfg.IMGSZ, cfg.IMGSZ, 3), dtype=np.uint8)
    try:
        model.predict(dummy, imgsz=cfg.IMGSZ, verbose=False, device=device)
        print("✅ OpenVINO model ready")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")

    if logger:
        logger.info(f"OpenVINO model: {cfg.OV_MODEL_PATH}")

else:
    print(f"\n📦 Loading PyTorch model: {cfg.MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH, task='detect')

    # Optional INT8 quantisation for Pi 5 CPU
    if hardware.IS_RASPBERRY_PI5 and not hardware.HAS_HAILO_NPU:
        try:
            model.model = torch.quantization.quantize_dynamic(
                model.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✅ Model quantized to INT8")
            if logger:
                logger.info("Model quantized to INT8")
        except Exception as e:
            print(f"⚠️  Quantization skipped: {e}")

    if logger:
        logger.info(f"PyTorch model: {cfg.MODEL_PATH}")

# ── Model info (safe after warmup) ────────────────────────────────────────────
try:
    class_names = list(model.names.values())
    num_classes = len(model.names)
except Exception:
    class_names = ['unknown']
    num_classes  = 0

print(f"\n📦 Model Info:")
print(f"   Classes    : {num_classes} → {class_names}")
print(f"   Conf       : {cfg.CONF_THRESH}")
print(f"   Img size   : {cfg.IMGSZ}")
print(f"   Backend    : {'OpenVINO' if cfg.USE_OV else 'PyTorch'}")
print(f"   Tracker    : {'disabled (OpenVINO)' if cfg.USE_OV else cfg.TRACKER_TYPE if cfg.ENABLE_TRACKING else 'disabled'}")
print(f"   Device     : {device}")
print(f"   Frame skip : every {cfg.FRAME_SKIP_N} frames" if cfg.USE_FRAME_SKIP else "   Frame skip : disabled")

if logger:
    logger.info(f"  Backend: {'OpenVINO' if cfg.USE_OV else 'PyTorch'}")
    logger.info(f"  Classes: {class_names}")
    logger.info(f"  Conf: {cfg.CONF_THRESH}, ImgSz: {cfg.IMGSZ}")
    logger.info(f"  Frame skip: {cfg.USE_FRAME_SKIP} (N={cfg.FRAME_SKIP_N})")
    logger.info(f"  Device: {device}")

# ── 5. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from video_processor import process_video
    process_video(model, device, logger)