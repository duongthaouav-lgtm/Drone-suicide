"""
hardware.py - Hardware detection and platform-specific configuration
Detects Raspberry Pi 5, Hailo NPU, GPU, and adjusts config accordingly.
"""
import os
import torch
import config as cfg


# =====================================================
# HARDWARE DETECTION
# =====================================================
def detect_raspberry_pi5() -> bool:
    """Detect if running on Raspberry Pi 5."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            return 'Pi 5' in model or 'Raspberry Pi 5' in model
    except Exception:
        return False


def detect_hailo_npu() -> bool:
    """Detect Hailo NPU (AI HAT)."""
    if os.path.exists('/dev/hailo0'):
        return True
    try:
        import hailo_platform  # noqa: F401
        return True
    except ImportError:
        return False


def get_system_memory_gb() -> float | None:
    """Return total system RAM in GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f.read().split('\n'):
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    return mem_kb / 1024 / 1024
    except Exception:
        pass
    return None


# =====================================================
# HARDWARE PROFILE
# =====================================================
IS_RASPBERRY_PI5 = detect_raspberry_pi5()
HAS_HAILO_NPU = detect_hailo_npu()
SYSTEM_MEMORY_GB = get_system_memory_gb()


def apply_hardware_overrides():
    """
    Mutate config values in-place based on detected hardware.
    Call once at startup before loading the model.
    """
    if IS_RASPBERRY_PI5:
        print("🍓 Raspberry Pi 5 detected - Optimizing for Pi 5 hardware...")

        if HAS_HAILO_NPU:
            print("🚀 AI HAT NPU (26TOPS) detected - NPU acceleration available!")
            cfg.CONF_THRESH = 0.15
            cfg.IMGSZ = 720
            cfg.ADAPTIVE_IMGSZ_MIN = 640
            cfg.ADAPTIVE_IMGSZ_MAX = 640
            cfg.ADAPTIVE_IMGSZ_BASE = 640
            print("⚙️  Config optimized for Pi 5 + NPU")
        else:
            print("⚠️  AI HAT NPU not detected - Using CPU inference")
            cfg.CONF_THRESH = 0.15
            cfg.IMGSZ = 512
            cfg.ADAPTIVE_IMGSZ_MIN = 512
            cfg.ADAPTIVE_IMGSZ_MAX = 512
            cfg.ADAPTIVE_IMGSZ_BASE = 512
            cfg.USE_HALF = False
            cfg.USE_KALMAN_FILTER = False
            cfg.USE_SMART_MULTI_SCALE = False
            cfg.TRACK_BUFFER = 30
            cfg.MAX_AGE = 20
            cfg.SMOOTHING_HISTORY = 2
            cfg.ENABLE_MEMORY_OPTIMIZATION = True
            cfg.USE_OV       = True    # Auto-enable OpenVINO on Pi 5 CPU
            cfg.USE_HALF     = False   # INT8 OpenVINO conflicts with FP16
            cfg.USE_FRAME_SKIP = True
            cfg.FRAME_SKIP_N   = 2
            print("⚙️  Config optimized for Pi 5 CPU (no NPU)")
            print("⚙️  OpenVINO + frame skip enabled for Pi 5")

        if SYSTEM_MEMORY_GB and SYSTEM_MEMORY_GB < 4:
            cfg.IMGSZ = 480
            cfg.ADAPTIVE_IMGSZ_MAX = 576
            print("⚙️  Further optimized for low RAM")

        if SYSTEM_MEMORY_GB:
            print(f"💾 System RAM: {SYSTEM_MEMORY_GB:.2f} GB")

        cfg.ENABLE_MEMORY_OPTIMIZATION = True


def select_device() -> str:
    """
    Choose the best available inference device and configure PyTorch accordingly.
    Returns the device string to pass to YOLO.
    """
    if HAS_HAILO_NPU:
        # Ultralytics does not natively support Hailo — fall back to CPU
        # For full NPU support, export the model to ONNX and use Hailo runtime.
        print("🚀 AI HAT NPU detected (using CPU with NPU-optimised settings)")
        print("   Note: export model to ONNX + Hailo runtime for true NPU inference")
        return 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        half_tag = " with FP16" if cfg.USE_HALF else ""
        print(f"✅ GPU detected{half_tag}: {gpu_name} ({gpu_mem:.2f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device

    # CPU fallback
    cfg.USE_HALF = False
    if IS_RASPBERRY_PI5:
        torch.set_num_threads(4)
        print(f"🍓 Raspberry Pi 5 CPU (4 threads)")
    else:
        print("⚠️  No GPU found - using CPU (slower)")
    return 'cpu'