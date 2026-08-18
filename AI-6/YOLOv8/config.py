"""
config.py - All configuration settings for YOLO Detection & Tracking System
"""
import logging

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = "../weights/best3_openvino_model" #"../weights/best_int8_openvino_model"
VIDEO_PATH = "/dev/video0"
OUTPUT_DIR = "../results"

# =====================================================
# INFERENCE SETTINGS
# =====================================================
CONF_THRESH = 0.15
IOU_THRESH = 0.60
IMGSZ = 512

# Adaptive Image Size
USE_ADAPTIVE_IMGSZ = True
ADAPTIVE_IMGSZ_MIN = 512
ADAPTIVE_IMGSZ_MAX = 512
ADAPTIVE_IMGSZ_BASE = 512

# =====================================================
# MODEL & DEVICE
# =====================================================
MODEL_SIZE = "nano"     # "auto", "nano", "small", "medium", "large", "custom"
USE_HALF = True         # FP16 (requires GPU or NPU)
USE_TENSORRT = False    # Requires model export first
BATCH_SIZE = 1

# =====================================================
# TRACKING
# =====================================================
ENABLE_TRACKING = True
TRACKER_TYPE = "bytetrack"   # "bytetrack", "botsort", "kcf", "hybrid"
TRACK_BUFFER = 50
TRACK_CONF_THRESH = 0.30
TRACK_IOU_THRESH = 0.50
MIN_TRACK_FRAMES = 2
MAX_AGE = 60

# =====================================================
# TEMPORAL SMOOTHING
# =====================================================
USE_TEMPORAL_SMOOTHING = False  # No track IDs with OpenVINO
SMOOTHING_ALPHA = 0.8
SMOOTHING_HISTORY = 5

# =====================================================
# KALMAN FILTER
# =====================================================
USE_KALMAN_FILTER = False   # No track IDs with OpenVINO
KALMAN_PROCESS_NOISE = 0.03
KALMAN_MEASUREMENT_NOISE = 0.3

# =====================================================
# ADAPTIVE CONFIDENCE
# =====================================================
USE_ADAPTIVE_CONF = False   # Fixed graph with OpenVINO
ADAPTIVE_CONF_MIN = 0.05
ADAPTIVE_CONF_MAX = 0.20
ADAPTIVE_CONF_STEP = 0.01

# =====================================================
# HUNGARIAN MATCHING
# =====================================================
USE_HUNGARIAN_MATCHING = False
HUNGARIAN_IOU_THRESH = 0.5

# =====================================================
# OCCLUSION HANDLING
# =====================================================
USE_OCCLUSION_HANDLING = False  # No track IDs with OpenVINO
OCCLUSION_THRESH = 0.3
OCCLUSION_BUFFER = 10

# =====================================================
# MULTI-SCALE DETECTION
# =====================================================
USE_MULTI_SCALE = False
USE_SMART_MULTI_SCALE = False  # Fixed input with OpenVINO
MULTI_SCALE_FACTORS = [0.8, 1.0, 1.2]
MULTI_SCALE_THRESHOLD = 2
MULTI_SCALE_FRAME_INTERVAL = 3

# =====================================================
# OPENVINO INFERENCE
# =====================================================
USE_OV = False             # Set True to use OpenVINO instead of PyTorch
# OpenVINO model path — export first (on desktop):
#   yolo export model=best.pt format=openvino opset=12
#   This creates: best_openvino_model
#   pip install openvino
OV_MODEL_PATH = "../weights/best3_openvino_model" #"../weights/best_int8_openvino_model"

# =====================================================
# FRAME SKIPPING
# =====================================================
USE_FRAME_SKIP = True        # Skip frames to boost FPS
FRAME_SKIP_N   = 3           # OpenVINO is slower on Pi 5, skip more aggressively
                             # ByteTrack fills in the gaps between detections
                             # Increase to 3 or 4 for more speed, less accuracy

# =====================================================
# ADVANCED OPTIMIZATIONS
# =====================================================
USE_PREPROCESSING_CACHE = True
USE_POSTPROCESSING_OPTIMIZATION = True
OPTIMIZE_GPU_MEMORY = False
USE_EARLY_EXIT = True
USE_BATCH_PROCESSING = False
ASYNC_INFERENCE = False

# =====================================================
# MEMORY MANAGEMENT
# =====================================================
ENABLE_MEMORY_OPTIMIZATION = False      # Overridden per hardware in setup.py
MEMORY_CLEANUP_INTERVAL = 60

# =====================================================
# CAMERA SETTINGS  (only used when VIDEO_PATH is a camera device)
# =====================================================
CAMERA_WIDTH  = 512    # Match IMGSZ to avoid resize overhead
CAMERA_HEIGHT = 512
CAMERA_FPS    = 40  

# =====================================================
# DISPLAY
# =====================================================
SHOW_FPS = True
DEBUG_DETECTIONS = False
SAVE_RESULT = False
WINDOW_NAME = "🚀 YOLO Segmentation Viewer"

# =====================================================
# PERFORMANCE MONITORING
# =====================================================
PERFORMANCE_MONITORING = True

# =====================================================
# LOGGING
# =====================================================
ENABLE_LOGGING = False
LOG_TO_FILE = True
LOG_LEVEL = logging.INFO
LOG_DETECTION_STATS = True
LOG_TRACKING_STATS = True
LOG_EVERY_N_FRAMES = 50