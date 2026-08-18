import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
import logging
from datetime import datetime
from collections import defaultdict
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
import platform
import gc
import threading
from queue import Queue
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction

try:
    from threading import Lock
    THREADING_AVAILABLE = True
except:
    THREADING_AVAILABLE = False

# Kalman Filter (optional - cần cài: pip install filterpy)
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("Warning: filterpy not installed. Kalman Filter disabled. Install with: pip install filterpy")

# =====================================================
# HARDWARE DETECTION - Raspberry Pi 5 + AI HAT
# =====================================================
def detect_raspberry_pi5():
    """Phát hiện Raspberry Pi 5"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'Pi 5' in model or 'Raspberry Pi 5' in model:
                return True
    except:
        pass
    return False

def detect_hailo_npu():
    """Phát hiện Hailo NPU (AI HAT)"""
    # Kiểm tra device file
    if os.path.exists('/dev/hailo0'):
        return True
    # Kiểm tra Hailo SDK
    try:
        import hailo_platform
        return True
    except ImportError:
        pass
    return False

def get_system_memory_gb():
    """Lấy thông tin RAM"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    return mem_kb / 1024 / 1024
    except:
        pass
    return None

# Phát hiện hardware
IS_RASPBERRY_PI5 = detect_raspberry_pi5()
HAS_HAILO_NPU = detect_hailo_npu()
SYSTEM_MEMORY_GB = get_system_memory_gb()

# Hailo NPU wrapper (nếu có)
HAILO_MODEL = None
HAILO_DEVICE = None
try:
    if HAS_HAILO_NPU:
        import hailo_platform
        # Sẽ được khởi tạo sau khi load model
        print("✅ Hailo Platform SDK available")
except ImportError:
    pass

if IS_RASPBERRY_PI5:
    print("🍓 Raspberry Pi 5 detected - Optimizing for Pi 5 hardware...")
    if HAS_HAILO_NPU:
        print("🚀 AI HAT NPU (26TOPS) detected - NPU acceleration available!")
    else:
        print("⚠️  AI HAT NPU not detected - Using CPU inference")
    if SYSTEM_MEMORY_GB:
        print(f"💾 System RAM: {SYSTEM_MEMORY_GB:.2f} GB")

# Performance monitoring
PERFORMANCE_MONITORING = True
PERF_STATS = {
    'inference_times': [],
    'preprocess_times': [],
    'postprocess_times': [],
    'total_times': [],
    'memory_usage': []
}

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "./weights/best copy.pt"      # model segmentation
VIDEO_PATH = "./video"            # đường dẫn video (hoặc 0 cho camera)
OUTPUT_DIR = "./results"              # thư mục lưu kết quả

# Model size selection (cho Pi 5)
MODEL_SIZE = "nano"  # "auto", "nano", "small", "medium", "large", "custom"
# Auto sẽ chọn dựa trên hardware:
# - Pi 5 CPU: nano hoặc small
# - Pi 5 + NPU: small hoặc medium
# - Desktop/GPU: medium hoặc large
# =====================================================
# CONFIG - TỐI ƯU HÓA
# =====================================================
# Tự động điều chỉnh cho Raspberry Pi 5
if IS_RASPBERRY_PI5:
    # Tối ưu cho Pi 5 - giảm image size để tăng FPS
    if HAS_HAILO_NPU:
        # NPU có thể xử lý tốt hơn
        CONF_THRESH = 0.15
        IMGSZ = 720 # Giảm từ 720 xuống 640 cho Pi 5
        ADAPTIVE_IMGSZ_MIN = 640  # Giảm cho Pi 5
        ADAPTIVE_IMGSZ_MAX = 832  # Giảm từ 832
        ADAPTIVE_IMGSZ_BASE = 640
        print("⚙️  Config optimized for Pi 5 + NPU")
    else:
        # CPU only - tối ưu mạnh hơn
        CONF_THRESH = 0.15
        IMGSZ = 576  # Giảm mạnh cho CPU
        ADAPTIVE_IMGSZ_MIN = 480  # Giảm cho CPU
        ADAPTIVE_IMGSZ_MAX = 640
        ADAPTIVE_IMGSZ_BASE = 576
        print("⚙️  Config optimized for Pi 5 CPU (no NPU)")
    
    # Tối ưu memory cho Pi 5
    if SYSTEM_MEMORY_GB and SYSTEM_MEMORY_GB < 4:
        IMGSZ = 480  # Giảm thêm nếu RAM thấp
        ADAPTIVE_IMGSZ_MAX = 576
        print("⚙️  Further optimized for low RAM")
else:
    # Config mặc định cho desktop/GPU
    CONF_THRESH = 0.15
    IMGSZ = 832
    ADAPTIVE_IMGSZ_MIN = 960
    ADAPTIVE_IMGSZ_MAX = 720
    ADAPTIVE_IMGSZ_BASE = 832

IOU_THRESH = 0.60  # IoU threshold cho NMS

# Adaptive Processing - TỐI ƯU ĐỂ TĂNG DETECTIONS
USE_ADAPTIVE_IMGSZ = True  # Tự động điều chỉnh image size
DEBUG_DETECTIONS = False  # Tắt debug để giảm noise
SAVE_RESULT = True
WINDOW_NAME = "🚀 YOLO Segmentation Viewer"

# FPS Optimization
USE_HALF = True  # FP16 để tăng tốc (cần GPU hoặc NPU)
USE_TENSORRT = False  # TensorRT optimization (cần export model trước)
BATCH_SIZE = 16  # Batch processing (tăng nếu GPU memory đủ)
ASYNC_INFERENCE = False  # Async processing (tăng FPS nhưng phức tạp hơn)

# Tắt FP16 trên Pi 5 CPU (không hỗ trợ)
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    USE_HALF = False

# Memory Management - Tối ưu cho Pi 5
ENABLE_MEMORY_OPTIMIZATION = IS_RASPBERRY_PI5
MEMORY_CLEANUP_INTERVAL = 60  # Cleanup memory mỗi N frames
if IS_RASPBERRY_PI5:
    print("⚙️  Memory optimization enabled for Pi 5")

# Detection Optimization
MULTI_SCALE = True  # Multi-scale detection (chính xác hơn nhưng chậm)
TEST_TIME_AUG = True  # Test Time Augmentation (chính xác hơn nhưng rất chậm)
# DETECT_EVERY_N_FRAMES = 1  # Đã tắt vì tracking cần detection mỗi frame

# Tracking Optimization
SHOW_FPS = True
ENABLE_TRACKING = True  # Bật tracking
TRACKER_TYPE = "bytetrack"  # "bytetrack", "botsort", "kcf", "hybrid"
TRACK_BUFFER = 50  # Tăng buffer để tracking tốt hơn
TRACK_CONF_THRESH = 0.30  # Confidence threshold riêng cho tracking
TRACK_IOU_THRESH = 0.50  # IoU threshold cho tracking
MIN_TRACK_FRAMES = 2  # Số frame tối thiểu để giữ track
MAX_AGE = 60  # Số frame tối đa để giữ track khi mất detection

# Tối ưu tracking cho Pi 5
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    # Giảm buffer và history để tiết kiệm memory
    TRACK_BUFFER = 30  # Giảm từ 50
    MAX_AGE = 20  # Giảm từ 30
    print("⚙️  Tracking optimized for Pi 5 CPU")

# Temporal Smoothing - TỐI ƯU ĐỂ TĂNG FPS
USE_TEMPORAL_SMOOTHING = True  # Làm mượt bounding boxes
SMOOTHING_ALPHA = 0.8  # Trọng số cho smoothing
SMOOTHING_HISTORY = 5  # Số frame để lưu history

# Giảm history cho Pi 5
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    SMOOTHING_HISTORY = 2  # Giảm từ 3 xuống 2

# Logging Configuration
ENABLE_LOGGING = True  # Bật logging
LOG_TO_FILE = True  # Lưu log vào file
LOG_LEVEL = logging.INFO  # DEBUG, INFO, WARNING, ERROR
LOG_DETECTION_STATS = True  # Log thống kê detections
LOG_TRACKING_STATS = True  # Log thống kê tracking
LOG_EVERY_N_FRAMES = 50  # Log mỗi N frames (0 = tắt)

# =====================================================
# SETUP LOGGING
# =====================================================
if ENABLE_LOGGING:
    # Tạo thư mục logs nếu chưa có
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Tạo tên file log với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"detection_log_{timestamp}.txt")
    
    # Setup logging với UTF-8 encoding để tránh lỗi emoji trên Windows
    handlers = []
    
    # Console handler với UTF-8
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    handlers.append(console_handler)
    
    # File handler với UTF-8 encoding
    if LOG_TO_FILE:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(LOG_LEVEL)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("YOLO Detection & Tracking System Started")
    logger.info("=" * 60)
else:
    logger = None

# =====================================================
# MODEL QUANTIZATION & OPTIMIZATION
# =====================================================
def optimize_model_for_pi5(model):
    """Tối ưu model cho Pi 5"""
    if IS_RASPBERRY_PI5:
        # Quantization cho Pi 5 CPU (INT8)
        try:
            if not HAS_HAILO_NPU:
                # Dynamic quantization cho CPU
                model.model = torch.quantization.quantize_dynamic(
                    model.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("✅ Model quantized to INT8 for Pi 5 CPU")
                if logger:
                    logger.info("Model quantized to INT8")
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            if logger:
                logger.warning(f"Quantization failed: {e}")
    return model

# =====================================================
# LOAD MODEL
# =====================================================
print(f"\n📦 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tối ưu model cho Pi 5
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    try:
        model = optimize_model_for_pi5(model)
    except Exception as e:
        print(f"⚠️  Model optimization skipped: {e}")

if logger:
    logger.info(f"Model loaded: {MODEL_PATH}")

# Hiển thị thông tin model
print(f"\n📦 Model Info:")  
print(f"   Classes: {len(model.names)}")
print(f"   Class names: {list(model.names.values())}")
print(f"   Confidence threshold: {CONF_THRESH}")
print(f"   Input size: {IMGSZ}")

if logger:
    logger.info(f"Model Info:")
    logger.info(f"   Classes: {len(model.names)}")
    logger.info(f"   Class names: {list(model.names.values())}")
    logger.info(f"   Confidence threshold: {CONF_THRESH}")
    logger.info(f"   Input size: {IMGSZ}")
    logger.info(f"   Tracking: {TRACKER_TYPE if ENABLE_TRACKING else 'Disabled'}")
    logger.info(f"   Temporal Smoothing: {USE_TEMPORAL_SMOOTHING}")

# Tự động detect device và tối ưu
if HAS_HAILO_NPU:
    # Ưu tiên NPU nếu có
    device = 'hailo'  # Sẽ được xử lý đặc biệt
    print(f"🚀 Đang sử dụng AI HAT NPU (26TOPS)")
    if logger:
        logger.info("Using AI HAT NPU (26TOPS)")
    # Note: Ultralytics có thể không hỗ trợ trực tiếp Hailo
    # Cần export model sang ONNX và sử dụng Hailo runtime
    # Tạm thời fallback về CPU nhưng với config tối ưu
    device = 'cpu'
    print("   Note: Using CPU with NPU-optimized settings")
    print("   For full NPU support, export model to ONNX and use Hailo runtime")
elif torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ Đang sử dụng GPU" + (" với FP16" if USE_HALF else ""))
    if logger:
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"   GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"   FP16: {USE_HALF}")
    # Tối ưu GPU memory
    torch.backends.cudnn.benchmark = True  # Tối ưu CUDNN
    torch.backends.cudnn.deterministic = False  # Tăng tốc (hy sinh reproducibility)
    if logger:
        logger.info("   CUDNN Benchmark: Enabled")
else:
    device = 'cpu'
    USE_HALF = False  # FP16 chỉ hoạt động trên GPU/NPU
    if IS_RASPBERRY_PI5:
        print(f"🍓 Đang sử dụng Raspberry Pi 5 CPU")
        # Tối ưu số threads cho Pi 5
        torch.set_num_threads(4)  # Pi 5 có 4 cores
        if logger:
            logger.info("Raspberry Pi 5 CPU detected")
            logger.info(f"   CPU Threads: {torch.get_num_threads()}")
    else:
        print(f"⚠️ Đang sử dụng CPU (chậm hơn GPU)")
        if logger:
            logger.warning("Using CPU (slower than GPU)")

# Khởi tạo KCF trackers nếu cần
kcf_trackers = {}  # Dictionary để lưu KCF trackers: {track_id: tracker}
kcf_boxes = {}  # Dictionary để lưu boxes của KCF: {track_id: box}

# Temporal smoothing history
tracking_history = {}  # {track_id: [boxes_history]}

# Kalman Filters cho mỗi track
kalman_filters = {}  # {track_id: KalmanFilter}

# Adaptive confidence tracking
current_adaptive_conf = CONF_THRESH
detection_quality_history = []  # Lưu quality scores

# Occlusion tracking
occluded_tracks = {}  # {track_id: frames_occluded}

# Frame skipping state
last_frame_results = None
skip_frame_counter = 0
tracking_stability = {}  # {track_id: stability_score}

# Adaptive processing state
adaptive_imgsz_history = []  # Lịch sử image sizes
scene_complexity = 0.5  # Độ phức tạp của scene (0-1)
last_detection_count = 0

# =====================================================
# ADVANCED TRACKING FEATURES
# =====================================================
# Kalman Filter cho Motion Prediction
USE_KALMAN_FILTER = True  # Bật Kalman Filter (sẽ tự động tắt nếu không có filterpy)
KALMAN_PROCESS_NOISE = 0.03  # Process noise
KALMAN_MEASUREMENT_NOISE = 0.3  # Measurement noise

# Tắt Kalman Filter trên Pi 5 CPU để tiết kiệm tài nguyên
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    USE_KALMAN_FILTER = False
    print("⚙️  Kalman Filter disabled for Pi 5 CPU (performance optimization)")

# Adaptive Confidence Threshold - TỐI ƯU ĐỂ TĂNG DETECTIONS
USE_ADAPTIVE_CONF = True  # Tự động điều chỉnh confidence
ADAPTIVE_CONF_MIN = 0.05  # Confidence tối thiểu (GIẢM từ 0.04 xuống 0.03 để detect nhiều hơn)
ADAPTIVE_CONF_MAX = 0.20  # Confidence tối đa (giảm từ 0.18 để phản ứng nhanh hơn)
ADAPTIVE_CONF_STEP = 0.01  # Bước điều chỉnh (TĂNG từ 0.008 lên 0.01 để phản ứng nhanh hơn khi thiếu detections)

# IoU-based Association với Hungarian Algorithm
USE_HUNGARIAN_MATCHING = True  # Sử dụng Hungarian algorithm
HUNGARIAN_IOU_THRESH = 0.5  # IoU threshold cho matching

# Occlusion Handling
USE_OCCLUSION_HANDLING = True  # Xử lý occlusion
OCCLUSION_THRESH = 0.3  # IoU threshold để phát hiện occlusion
OCCLUSION_BUFFER = 10  # Số frame để giữ track khi bị occluded

# Multi-scale Detection - TỐI ƯU: chỉ dùng khi cần nhưng phản ứng nhanh hơn
USE_MULTI_SCALE = False  # Multi-scale detection (chậm nhưng chính xác hơn)
USE_SMART_MULTI_SCALE = True  # Chỉ dùng multi-scale khi detect ít objects hoặc objects nhỏ
MULTI_SCALE_FACTORS = [0.8, 1.0, 1.2]  # Các scale factors
MULTI_SCALE_THRESHOLD = 2  # Số detections tối đa để trigger multi-scale
MULTI_SCALE_FRAME_INTERVAL = 3  # Dùng multi-scale mỗi N frames

# Tắt multi-scale trên Pi 5 CPU
if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
    USE_SMART_MULTI_SCALE = False
    print("⚙️  Multi-scale detection disabled for Pi 5 CPU")

# Smart Frame Skipping - TẮT ĐỂ TĂNG DETECTIONS
USE_SMART_SKIP = False  # TẮT frame skipping để xử lý tất cả frames (tăng detections)
SKIP_THRESHOLD_STABLE = 0.98  # Tracking stability threshold (không dùng khi tắt skip)
MAX_SKIP_FRAMES = 0  # Tắt skip hoàn toàn để detect mọi frame
SKIP_ON_NO_DETECTIONS = False  # Tắt skip khi không có detections
SKIP_CONSECUTIVE_NO_DET = 10  # Không dùng khi tắt skip

# Batch Processing - Tối ưu
USE_BATCH_PROCESSING = False  # Xử lý nhiều frames cùng lúc (tăng FPS nhưng tốn memory)
BATCH_SIZE = 1  # Số frames trong một batch (giảm để tránh OOM)

# Advanced Optimizations
USE_PREPROCESSING_CACHE = True  # Cache preprocessing để tăng tốc
USE_POSTPROCESSING_OPTIMIZATION = True  # Tối ưu post-processing
OPTIMIZE_GPU_MEMORY = True  # Tối ưu GPU memory usage
USE_EARLY_EXIT = True  # Early exit khi không có objects

# Thông báo tracker đang sử dụng
if ENABLE_TRACKING:
    if TRACKER_TYPE == "hybrid":
        tracker_info = "ByteTrack + KCF (Hybrid Mode)"
    elif TRACKER_TYPE == "botsort":
        tracker_info = "BotSORT"
    elif TRACKER_TYPE == "kcf":
        tracker_info = "KCF only"
    else:
        tracker_info = "ByteTrack"
    print(f"🎯 Tracking: {tracker_info}")
    if logger:
        logger.info(f"Tracking: {tracker_info}")
        logger.info(f"   Track Buffer: {TRACK_BUFFER}")
        logger.info(f"   Min Track Frames: {MIN_TRACK_FRAMES}")
        logger.info(f"   Max Age: {MAX_AGE}")
        
        # Log advanced features
        logger.info("Advanced Features:")
        logger.info(f"   Kalman Filter: {USE_KALMAN_FILTER and KALMAN_AVAILABLE}")
        logger.info(f"   Adaptive Confidence: {USE_ADAPTIVE_CONF}")
        logger.info(f"   Hungarian Matching: {USE_HUNGARIAN_MATCHING}")
        logger.info(f"   Occlusion Handling: {USE_OCCLUSION_HANDLING}")
        logger.info(f"   Multi-scale Detection: {USE_MULTI_SCALE}")
        logger.info(f"   Smart Frame Skip: {USE_SMART_SKIP}")
        logger.info(f"   Batch Processing: {USE_BATCH_PROCESSING}")
        logger.info(f"   Temporal Smoothing: {USE_TEMPORAL_SMOOTHING}")
else:
    print(f"🎯 Tracking: Disabled")
    if logger:
        logger.info("Tracking: Disabled")

# =====================================================
# HELPER FUNCTIONS CHO ADVANCED FEATURES
# =====================================================

def process_single_video(cap, video_name):
    global current_adaptive_conf, detection_quality_history
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print(f"📹 {video_name}: {width}x{height} @ {fps}fps")

    frame_count = 0
    current_fps = 0
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        start = time.time()
        frame, stats = process_frame(frame, frame_count, 0, current_fps)

        # FPS calc
        t = time.time() - start
        frame_times.append(t)
        if len(frame_times) > 30:
            frame_times.pop(0)
        current_fps = 1.0 / (sum(frame_times)/len(frame_times))

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

def create_kalman_filter(x, y, w, h):
    """Tạo Kalman Filter cho một track"""
    if not KALMAN_AVAILABLE:
        return None
    kf = KalmanFilter(dim_x=8, dim_z=4)  # 8 states (x,y,w,h,vx,vy,vw,vh), 4 measurements
    
    # State: [x, y, w, h, vx, vy, vw, vh]
    kf.x = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
        [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
        [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
        [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
        [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
        [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
        [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
        [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
    ], dtype=np.float32)
    
    # Measurement matrix (chỉ đo x, y, w, h)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ], dtype=np.float32)
    
    # Covariance matrices
    kf.P *= 1000.  # Uncertainty
    kf.R = np.eye(4) * KALMAN_MEASUREMENT_NOISE  # Measurement noise
    kf.Q = np.eye(8) * KALMAN_PROCESS_NOISE  # Process noise
    
    return kf

def update_kalman_filter(kf, x, y, w, h):
    """Update Kalman Filter với measurement mới"""
    if kf is None or not KALMAN_AVAILABLE:
        return np.array([x, y, w, h], dtype=np.float32)
    kf.predict()
    kf.update(np.array([x, y, w, h], dtype=np.float32))
    return kf.x[:4]  # Trả về predicted position

def calculate_iou(box1, box2):
    """Tính IoU giữa 2 boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Tính intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def hungarian_matching(tracks, detections, iou_thresh=0.5):
    """Sử dụng Hungarian Algorithm để match tracks với detections"""
    if len(tracks) == 0 or len(detections) == 0:
        return []
    
    # Tạo cost matrix (1 - IoU)
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track_box in enumerate(tracks):
        for j, det_box in enumerate(detections):
            iou = calculate_iou(track_box, det_box)
            cost_matrix[i, j] = 1.0 - iou  # Cost = 1 - IoU
    
    # Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Lọc matches dựa trên IoU threshold
    matches = []
    for i, j in zip(row_indices, col_indices):
        iou = 1.0 - cost_matrix[i, j]
        if iou >= iou_thresh:
            matches.append((i, j, iou))
    
    return matches

def adaptive_confidence_adjustment(detection_quality):
    """Điều chỉnh confidence threshold dựa trên detection quality - TỐI ƯU ĐỂ TĂNG DETECTIONS"""
    global current_adaptive_conf
    
    if not USE_ADAPTIVE_CONF or len(detection_quality) < 2:  # Giảm từ 3 xuống 2 để phản ứng nhanh hơn
        return current_adaptive_conf
    
    # Tính quality trung bình và tỷ lệ frames có detections
    recent_quality = detection_quality[-8:]  # GIẢM từ 12 xuống 8 để phản ứng NHANH HƠN
    avg_quality = np.mean(recent_quality)
    detection_rate = sum(1 for q in recent_quality if q > 0) / len(recent_quality) if len(recent_quality) > 0 else 0
    
    # Điều chỉnh confidence - ƯU TIÊN TĂNG DETECTIONS
    # Nếu detection rate < 60%, giảm threshold MẠNH (tăng từ 55% lên 60%)
    if detection_rate < 0.60:  # Detection rate thấp -> GIẢM THRESHOLD MẠNH
        current_adaptive_conf = max(ADAPTIVE_CONF_MIN, current_adaptive_conf - ADAPTIVE_CONF_STEP * 2.5)  # Giảm nhanh hơn (tăng từ 2 lên 2.5)
    elif detection_rate < 0.70:  # Detection rate trung bình -> giảm nhẹ
        current_adaptive_conf = max(ADAPTIVE_CONF_MIN, current_adaptive_conf - ADAPTIVE_CONF_STEP)
    elif avg_quality < 0.5:  # Quality thấp -> giảm threshold (tăng từ 0.4 lên 0.5)
        current_adaptive_conf = max(ADAPTIVE_CONF_MIN, current_adaptive_conf - ADAPTIVE_CONF_STEP * 0.5)
    elif avg_quality > 0.85:  # Quality cao -> tăng threshold (tăng từ 0.8 lên 0.85 để chỉ tăng khi rất tốt)
        current_adaptive_conf = min(ADAPTIVE_CONF_MAX, current_adaptive_conf + ADAPTIVE_CONF_STEP)
    
    return current_adaptive_conf

def adaptive_imgsz_adjustment(detection_count, last_detection_count):
    """Điều chỉnh image size dựa trên số lượng detections"""
    global adaptive_imgsz_history, scene_complexity
    
    if not USE_ADAPTIVE_IMGSZ:
        return IMGSZ
    
    # Cập nhật scene complexity
    if detection_count == 0 and last_detection_count == 0:
        # Không có detections -> giảm size để tăng tốc
        scene_complexity = max(0.0, scene_complexity - 0.05)
    elif detection_count > 0:
        # Có detections -> tăng complexity
        scene_complexity = min(1.0, scene_complexity + 0.02)
    
    # Điều chỉnh image size dựa trên complexity
    if scene_complexity < 0.3:
        # Scene đơn giản -> dùng size nhỏ
        imgsz = ADAPTIVE_IMGSZ_MIN
    elif scene_complexity > 0.7:
        # Scene phức tạp -> dùng size lớn
        imgsz = ADAPTIVE_IMGSZ_MAX
    else:
        # Scene trung bình -> dùng size base
        imgsz = ADAPTIVE_IMGSZ_BASE
    
    # Lưu history
    adaptive_imgsz_history.append(imgsz)
    if len(adaptive_imgsz_history) > 10:
        adaptive_imgsz_history.pop(0)
    
    return imgsz

def detect_occlusion(boxes, iou_thresh=0.3):
    """Phát hiện occlusion dựa trên IoU giữa các boxes"""
    occluded = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_thresh:
                # Box nhỏ hơn bị occluded bởi box lớn hơn
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                if area_i < area_j:
                    occluded.add(i)
                else:
                    occluded.add(j)
    return occluded

# =====================================================
# MEMORY MANAGEMENT
# =====================================================
def cleanup_memory():
    """Dọn dẹp memory cho Pi 5"""
    if ENABLE_MEMORY_OPTIMIZATION:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# =====================================================
# PERFORMANCE MONITORING
# =====================================================
def update_perf_stats(inference_time, preprocess_time=0, postprocess_time=0):
    """Cập nhật performance statistics"""
    if PERFORMANCE_MONITORING:
        total_time = inference_time + preprocess_time + postprocess_time
        PERF_STATS['inference_times'].append(inference_time)
        PERF_STATS['preprocess_times'].append(preprocess_time)
        PERF_STATS['postprocess_times'].append(postprocess_time)
        PERF_STATS['total_times'].append(total_time)
        
        # Giữ tối đa 100 samples
        for key in PERF_STATS:
            if len(PERF_STATS[key]) > 100:
                PERF_STATS[key].pop(0)

def get_perf_summary():
    """Lấy thống kê performance"""
    if not PERFORMANCE_MONITORING or len(PERF_STATS['inference_times']) == 0:
        return None
    
    summary = {
        'avg_inference_ms': np.mean(PERF_STATS['inference_times']) * 1000,
        'avg_total_ms': np.mean(PERF_STATS['total_times']) * 1000,
        'min_inference_ms': np.min(PERF_STATS['inference_times']) * 1000,
        'max_inference_ms': np.max(PERF_STATS['inference_times']) * 1000,
        'fps': 1.0 / np.mean(PERF_STATS['total_times']) if np.mean(PERF_STATS['total_times']) > 0 else 0
    }
    return summary

# =====================================================
# XỬ LÝ FRAME VỚI TRACKING (TỐI ƯU) + KCF
# =====================================================
def process_frame(frame, frame_count=0, total_frames=0, fps=0):
    """Xử lý một frame từ video với tracking - tối ưu cho tốc độ và accuracy"""
    global kcf_trackers, kcf_boxes, tracking_history
    global kalman_filters, current_adaptive_conf, detection_quality_history
    global occluded_tracks, last_frame_results, skip_frame_counter, tracking_stability
    
    # Performance monitoring
    frame_start_time = time.time()
    preprocess_time = 0
    
    # Adaptive Confidence Threshold
    conf_thresh_to_use = CONF_THRESH
    if USE_ADAPTIVE_CONF:
        conf_thresh_to_use = adaptive_confidence_adjustment(detection_quality_history)
        if ENABLE_TRACKING:
            conf_thresh_to_use = max(conf_thresh_to_use, TRACK_CONF_THRESH)
    
    # Adaptive Image Size - TỐI ƯU ĐỂ TĂNG DETECTIONS
    imgsz_to_use = IMGSZ  # Dùng IMGSZ mặc định (720) để cân bằng
    if USE_ADAPTIVE_IMGSZ:
        # ƯU TIÊN TĂNG DETECTIONS - phản ứng nhanh hơn
        if len(detection_quality_history) > 3:  # Giảm từ 5 xuống 3 để phản ứng nhanh hơn
            recent_no_detections = sum(1 for q in detection_quality_history[-8:] if q == 0)  # Giảm từ 10 xuống 8
            detection_rate = 1.0 - (recent_no_detections / min(8, len(detection_quality_history)))
            
            # Nếu detection rate < 50%, tăng size để detect tốt hơn (tăng từ 40% lên 50%)
            if detection_rate < 0.50:
                imgsz_to_use = ADAPTIVE_IMGSZ_MAX  # Dùng size lớn (832) để detect tốt hơn
            elif detection_rate > 0.75:  # Tăng từ 0.7 lên 0.75 để chỉ giảm size khi rất tốt
                imgsz_to_use = ADAPTIVE_IMGSZ_MIN  # Dùng size nhỏ (640) để tăng FPS
            else:
                imgsz_to_use = ADAPTIVE_IMGSZ_BASE  # Dùng size base (720)
        else:
            imgsz_to_use = ADAPTIVE_IMGSZ_BASE  # Mặc định dùng size base
    
    # Smart Multi-scale Detection - TỐI ƯU: trigger sớm hơn khi thiếu detections
    use_multi_scale_now = False
    if USE_SMART_MULTI_SCALE and len(detection_quality_history) > 0:
        recent_detections = sum(1 for q in detection_quality_history[-5:] if q > 0)
        # Dùng multi-scale mỗi N frames để giảm overhead
        if frame_count % MULTI_SCALE_FRAME_INTERVAL == 0 and recent_detections <= MULTI_SCALE_THRESHOLD:
            # Nếu ít detections, có thể có objects nhỏ -> dùng multi-scale
            use_multi_scale_now = True
            imgsz_to_use = int(imgsz_to_use * 1.15)  # Tăng size 15% (tăng từ 10% để detect tốt hơn)
    
    # Performance monitoring - inference time
    inference_start = time.time()
    
    # Giảm max_det cho Pi 5 để tiết kiệm memory
    max_detections = 500  # Mặc định cho các thiết bị mạnh hơn
    if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
        max_detections = 150  # Giảm cho Pi 5 CPU
    elif IS_RASPBERRY_PI5 and HAS_HAILO_NPU:
        max_detections = 250  # NPU có thể xử lý nhiều hơn

    # Pre-inference enhancement options (unsharp mask / CLAHE)
    # Default: disabled to avoid degrading detection. Toggle flags below to enable.
    # try:
    #     use_sharpen_for_inference = False
    #     use_clahe_for_inference = False

    #     # CLAHE enhancement (good for contrast without sharpening)
    #     if use_clahe_for_inference:
    #         lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #         l, a, b = cv2.split(lab)
    #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #         l = clahe.apply(l)
    #         lab = cv2.merge((l, a, b))
    #         clahe_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #     else:
    #         clahe_frame = frame

    #     # Unsharp mask (sharpen)
    #     blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    #     # Use mild weights to avoid artifacts (adjust as needed)
    #     sharpened = cv2.addWeighted(frame, 1.2, blurred, -0.2, 0)

    #     # Choose which processed frame to run inference on
    #     if use_sharpen_for_inference:
    #         inference_frame = sharpened
    #     else:
    #         inference_frame = clahe_frame
    # except Exception:
    #     inference_frame = frame
    
    # Sử dụng track() nếu bật tracking, nếu không dùng predict()
    if ENABLE_TRACKING and TRACKER_TYPE in ["bytetrack", "botsort", "hybrid"]:
        tracker_name = "botsort.yaml" if TRACKER_TYPE == "botsort" else "bytetrack.yaml"
        results = model.track(
            frame,
            # inference_frame,
            conf=conf_thresh_to_use, 
            iou=TRACK_IOU_THRESH if ENABLE_TRACKING else IOU_THRESH,
            imgsz=imgsz_to_use, 
            verbose=DEBUG_DETECTIONS,
            device=device,
            half=USE_HALF,
            persist=True,
            tracker=tracker_name,
            show=False,
            agnostic_nms=False,
            max_det=max_detections,
            classes=None,  # Detect tất cả classes
            retina_masks=False,  # Tắt để tăng tốc
            stream=False,
            # visualize=True,
        )
    else:
        results = model.predict(
            frame,
            # inference_frame,
            conf=conf_thresh_to_use,
            iou=IOU_THRESH,
            imgsz=imgsz_to_use, 
            verbose=DEBUG_DETECTIONS,
            device=device,
            augment=False,
            half=USE_HALF,
            show=False,
            agnostic_nms=False,
            max_det=max_detections,
            classes=None,
            retina_masks=False,
            stream=False,
            # visualize=True,
        )
    
    # Performance monitoring
    inference_time = time.time() - inference_start
    postprocess_start = time.time()
    
    names = model.names
    
    # Debug: Kiểm tra raw detections trước khi filter
    raw_detection_count = 0
    for r in results:
        if r.boxes is not None:
            raw_detection_count += len(r.boxes)
    
    # Early Exit Optimization - Nếu không có detections và đã skip nhiều lần
    if USE_EARLY_EXIT and raw_detection_count == 0 and skip_frame_counter > 0:
        # Có thể skip processing nếu không có detections
        pass
    
    # Tạo bản sao để vẽ overlay
    annotated_frame = frame.copy()
    detection_count = 0
    track_count = 0
    class_counts = defaultdict(int)  # Đếm số lượng mỗi class
    track_ids_set = set()  # Lưu track IDs

    for r in results:
        # Lấy track IDs nếu có (lấy trước để dùng cho cả boxes và masks)
        track_ids = None
        if ENABLE_TRACKING and r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            track_count = len(track_ids)
        
        # Boxes với tracking
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_data = r.boxes.xyxy.cpu().numpy()  # Lấy tất cả boxes cùng lúc
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            
            # KCF Tracking bổ sung (nếu dùng hybrid mode)
            if TRACKER_TYPE == "hybrid" and track_ids is not None:
                # Cập nhật hoặc tạo KCF trackers mới
                active_track_ids = set(track_ids)
                
                # Xóa trackers không còn active
                for tid in list(kcf_trackers.keys()):
                    if tid not in active_track_ids:
                        del kcf_trackers[tid]
                        if tid in kcf_boxes:
                            del kcf_boxes[tid]
                
                # Cập nhật hoặc tạo mới KCF trackers
                for i, (box, track_id) in enumerate(zip(boxes_data, track_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    
                    if track_id not in kcf_trackers:
                        # Tạo KCF tracker mới
                        kcf_tracker = cv2.TrackerKCF_create()
                        bbox = (x1, y1, w, h)
                        kcf_tracker.init(frame, bbox)
                        kcf_trackers[track_id] = kcf_tracker
                        kcf_boxes[track_id] = (x1, y1, x2, y2)
                    else:
                        # Cập nhật KCF tracker
                        try:
                            success, bbox = kcf_trackers[track_id].update(frame)
                            if success:
                                kx1, ky1, kw, kh = map(int, bbox)
                                kx2, ky2 = kx1 + kw, ky1 + kh
                                # Kết hợp với YOLO detection (weighted average)
                                alpha = 0.7  # Trọng số cho YOLO
                                kcf_boxes[track_id] = (
                                    int(alpha * x1 + (1 - alpha) * kx1),
                                    int(alpha * y1 + (1 - alpha) * ky1),
                                    int(alpha * x2 + (1 - alpha) * kx2),
                                    int(alpha * y2 + (1 - alpha) * ky2)
                                )
                            else:
                                # KCF failed, dùng YOLO box
                                kcf_boxes[track_id] = (x1, y1, x2, y2)
                        except:
                            # Nếu KCF lỗi, dùng YOLO box
                            kcf_boxes[track_id] = (x1, y1, x2, y2)
            
            # Occlusion Detection
            if USE_OCCLUSION_HANDLING and len(boxes_data) > 1:
                occluded_indices = detect_occlusion(boxes_data, OCCLUSION_THRESH)
            else:
                occluded_indices = set()
            
            for i, (box, conf, cls) in enumerate(zip(boxes_data, confs, clss)):
                x1, y1, x2, y2 = map(int, box)
                label = names[cls]
                w, h = x2 - x1, y2 - y1
                
                # Kalman Filter cho Motion Prediction
                if USE_KALMAN_FILTER and KALMAN_AVAILABLE and track_ids is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    if track_id not in kalman_filters:
                        # Tạo Kalman Filter mới
                        kf = create_kalman_filter(x1 + w/2, y1 + h/2, w, h)
                        if kf is not None:
                            kalman_filters[track_id] = kf
                    else:
                        # Update và predict với Kalman Filter
                        try:
                            kf = kalman_filters[track_id]
                            if kf is not None:
                                predicted = update_kalman_filter(kf, x1 + w/2, y1 + h/2, w, h)
                                # Sử dụng predicted position (weighted với measurement)
                                kf_weight = 0.3  # Trọng số cho Kalman prediction
                                x1 = int((1 - kf_weight) * x1 + kf_weight * (predicted[0] - predicted[2]/2))
                                y1 = int((1 - kf_weight) * y1 + kf_weight * (predicted[1] - predicted[3]/2))
                                x2 = int((1 - kf_weight) * x2 + kf_weight * (predicted[0] + predicted[2]/2))
                                y2 = int((1 - kf_weight) * y2 + kf_weight * (predicted[1] + predicted[3]/2))
                        except Exception as e:
                            if logger:
                                logger.debug(f"Kalman Filter error: {e}")
                            pass  # Nếu Kalman Filter lỗi, dùng detection gốc
                
                # Temporal Smoothing - làm mượt bounding boxes
                if USE_TEMPORAL_SMOOTHING and track_ids is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    if track_id not in tracking_history:
                        tracking_history[track_id] = []
                    tracking_history[track_id].append((x1, y1, x2, y2))
                    # Giữ tối đa SMOOTHING_HISTORY frames
                    if len(tracking_history[track_id]) > SMOOTHING_HISTORY:
                        tracking_history[track_id].pop(0)
                    # Tính trung bình có trọng số
                    if len(tracking_history[track_id]) > 1:
                        weights = np.exp(np.linspace(-1, 0, len(tracking_history[track_id])))
                        weights = weights / weights.sum()
                        boxes_array = np.array(tracking_history[track_id])
                        x1 = int(np.average(boxes_array[:, 0], weights=weights))
                        y1 = int(np.average(boxes_array[:, 1], weights=weights))
                        x2 = int(np.average(boxes_array[:, 2], weights=weights))
                        y2 = int(np.average(boxes_array[:, 3], weights=weights))
                
                # Occlusion Handling
                if USE_OCCLUSION_HANDLING and track_ids is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    if i in occluded_indices:
                        occluded_tracks[track_id] = occluded_tracks.get(track_id, 0) + 1
                        # Nếu bị occluded quá lâu, đánh dấu
                        if occluded_tracks[track_id] > OCCLUSION_BUFFER:
                            continue  # Skip vẽ box này
                    else:
                        occluded_tracks[track_id] = 0
                
                # Sử dụng KCF box nếu có (hybrid mode)
                if TRACKER_TYPE == "hybrid" and track_ids is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    if track_id in kcf_boxes:
                        kx1, ky1, kx2, ky2 = kcf_boxes[track_id]
                        # Weighted average với YOLO
                        x1 = int(SMOOTHING_ALPHA * x1 + (1 - SMOOTHING_ALPHA) * kx1)
                        y1 = int(SMOOTHING_ALPHA * y1 + (1 - SMOOTHING_ALPHA) * ky1)
                        x2 = int(SMOOTHING_ALPHA * x2 + (1 - SMOOTHING_ALPHA) * kx2)
                        y2 = int(SMOOTHING_ALPHA * y2 + (1 - SMOOTHING_ALPHA) * ky2)
                
                # Màu sắc dựa trên track ID (nếu có) để dễ phân biệt
                if track_ids is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    # Tạo màu dựa trên track ID
                    color_id = track_id % 255
                    color = (
                        int(255 * np.sin(color_id * 0.1) ** 2),
                        int(255 * np.sin(color_id * 0.1 + 2) ** 2),
                        int(255 * np.sin(color_id * 0.1 + 4) ** 2)
                    )
                else:
                    color = (0, 255, 0)  # Màu xanh lá mặc định
                
                # Vẽ bounding box (dày hơn nếu có tracking, đặc biệt nếu bị occluded)
                thickness = 3 if track_ids is not None else 2
                if USE_OCCLUSION_HANDLING and track_ids is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    if occluded_tracks.get(track_id, 0) > 0:
                        thickness = 2  # Mỏng hơn khi bị occluded
                        color = tuple(int(c * 0.5) for c in color)  # Mờ hơn
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Vẽ label với track ID và tracker type
                if track_ids is not None and i < len(track_ids):
                    tracker_label = "KCF" if TRACKER_TYPE == "hybrid" else "BT"
                    label_text = f"ID:{track_ids[i]}[{tracker_label}] {label} {conf:.2f}"
                else:
                    label_text = f"{label} {conf:.2f}"
                
                # Vẽ background cho label
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - text_height - 8), 
                            (x1 + text_width + 4, y1), 
                            color, -1)
                cv2.putText(annotated_frame, label_text, 
                          (x1 + 2, max(y1 - 5, text_height)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                detection_count += 1
                class_counts[label] += 1
                if track_ids is not None and i < len(track_ids):
                    track_ids_set.add(track_ids[i])

        # Masks (Segmentation) - Tối ưu
        # if r.masks is not None:
        #     # Sử dụng polygon coordinates thay vì resize mask (nhanh hơn)
        #     if hasattr(r.masks, 'xy') and r.masks.xy is not None:
        #         for idx, mask_poly in enumerate(r.masks.xy):
        #             if len(mask_poly) > 0:
        #                 pts = np.int32([mask_poly])
                        
        #                 # Màu mask dựa trên track ID nếu có (match với box index)
        #                 if track_ids is not None and idx < len(track_ids):
        #                     color_id = track_ids[idx] % 255
        #                     mask_color = (
        #                         int(255 * np.sin(color_id * 0.1) ** 2),
        #                         int(255 * np.sin(color_id * 0.1 + 2) ** 2),
        #                         int(255 * np.sin(color_id * 0.1 + 4) ** 2)
        #                     )
        #                 else:
        #                     mask_color = (0, 0, 255)  # Màu đỏ mặc định
                        
        #                 # Vẽ filled polygon với độ trong suốt
        #                 overlay = annotated_frame.copy()
        #                 cv2.fillPoly(overlay, [pts], mask_color)
        #                 annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
        #                 # Vẽ contour
        #                 cv2.polylines(annotated_frame, [pts], True, mask_color, 2)
        #     else:
        #         # Fallback: xử lý mask data nếu không có polygon
        #         masks = r.masks.data.cpu().numpy()
        #         for idx, mask in enumerate(masks):
        #             mask_resized = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
        #             mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
        #             # Màu mask dựa trên track ID
        #             if track_ids is not None and idx < len(track_ids):
        #                 color_id = track_ids[idx] % 255
        #                 mask_color = (
        #                     int(255 * np.sin(color_id * 0.1) ** 2),
        #                     int(255 * np.sin(color_id * 0.1 + 2) ** 2),
        #                     int(255 * np.sin(color_id * 0.1 + 4) ** 2)
        #                 )
        #             else:
        #                 mask_color = (0, 0, 255)
                    
        #             overlay = annotated_frame.copy()
        #             overlay[mask_binary == 1] = mask_color
        #             annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

    # Overlay info với FPS, tracking và device
    tracker_name = TRACKER_TYPE.upper() if ENABLE_TRACKING else "NONE"
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    info_text = f"Frame: {frame_count}/{total_frames} | Det: {detection_count}"
    if DEBUG_DETECTIONS and raw_detection_count > detection_count:
        info_text += f" (Raw: {raw_detection_count})"
    if ENABLE_TRACKING and track_count > 0:
        info_text += f" | Tracks: {track_count} [{tracker_name}]"
    if SHOW_FPS:
        info_text += f" | FPS: {fps:.1f}" if fps > 0 else " | FPS: --"
    if USE_ADAPTIVE_CONF:
        info_text += f" | Conf: {current_adaptive_conf:.2f}"
    info_text += f" | {device_name}"
    if total_frames == 0:
        info_text = f"Frame: Live | Det: {detection_count}"
        if DEBUG_DETECTIONS and raw_detection_count > detection_count:
            info_text += f" (Raw: {raw_detection_count})"
        if ENABLE_TRACKING and track_count > 0:
            info_text += f" | Tracks: {track_count} [{tracker_name}]"
        if SHOW_FPS:
            info_text += f" | FPS: {fps:.1f}" if fps > 0 else " | FPS: --"
        if USE_ADAPTIVE_CONF:
            info_text += f" | Conf: {current_adaptive_conf:.2f}"
        info_text += f" | {device_name}"
    
    # Logging thông tin frame
    if logger and LOG_EVERY_N_FRAMES > 0 and frame_count % LOG_EVERY_N_FRAMES == 0:
        log_msg = f"Frame {frame_count}/{total_frames}: Det={detection_count}, Tracks={track_count}, FPS={fps:.1f}"
        if class_counts:
            classes_str = ", ".join([f"{k}:{v}" for k, v in class_counts.items()])
            log_msg += f" | Classes: {classes_str}"
        logger.info(log_msg)
    
    # Debug warning nếu không có detections
    if DEBUG_DETECTIONS and detection_count == 0 and frame_count % 30 == 0:
        warning_msg = f"⚠️ Frame {frame_count}: Không có detections (Conf: {CONF_THRESH}, ImgSz: {IMGSZ})"
        print(warning_msg)
        if logger:
            logger.warning(warning_msg)
    
    # Vẽ background cho text (tối ưu)
    (text_width, text_height), _ = cv2.getTextSize(
        info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(annotated_frame, (10, 10), (20 + text_width, 40), (0, 0, 0), -1)
    cv2.putText(annotated_frame, info_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Cập nhật detection quality cho adaptive confidence
    if USE_ADAPTIVE_CONF:
        # Quality score: 1.0 nếu có detections, 0.0 nếu không có
        quality_score = 1.0 if detection_count > 0 else 0.0
        # Nếu có raw detections nhưng không pass threshold, quality = 0.1 (GIẢM để trigger mạnh hơn)
        if raw_detection_count > 0 and detection_count == 0:
            quality_score = 0.1  # GIẢM từ 0.2 xuống 0.1 để trigger giảm threshold MẠNH HƠN
        # Nếu có nhiều raw detections nhưng ít final detections, cần giảm threshold
        elif raw_detection_count > detection_count * 1.5:  # Giảm từ 2 xuống 1.5 để nhạy hơn
            quality_score = 0.3  # GIẢM từ 0.4 xuống 0.3 để trigger giảm threshold
        
        detection_quality_history.append(quality_score)
        if len(detection_quality_history) > 30:  # Giảm từ 40 xuống 30 để phản ứng nhanh hơn
            detection_quality_history.pop(0)
    
    # Cleanup Kalman filters không còn active
    if USE_KALMAN_FILTER and track_ids_set:
        active_track_ids = set(track_ids_set)
        for tid in list(kalman_filters.keys()):
            if tid not in active_track_ids:
                del kalman_filters[tid]
    
    # Performance monitoring - postprocess time
    postprocess_time = time.time() - postprocess_start
    total_time = time.time() - frame_start_time
    update_perf_stats(inference_time, preprocess_time, postprocess_time)
    
    # Memory cleanup định kỳ cho Pi 5
    if ENABLE_MEMORY_OPTIMIZATION and frame_count > 0 and frame_count % MEMORY_CLEANUP_INTERVAL == 0:
        cleanup_memory()
    
    # Return stats để cập nhật thống kê
    stats = {
        'detection_count': detection_count,
        'track_count': track_count,
        'class_counts': dict(class_counts),
        'track_ids': list(track_ids_set),
        'inference_time_ms': inference_time * 1000,
        'total_time_ms': total_time * 1000
    }
    
    return annotated_frame, stats

# =====================================================
# XỬ LÝ VIDEO
# =====================================================
def process_video():
    """Xử lý video file hoặc camera"""
    # Hỗ trợ camera (0 = default camera)
    if VIDEO_PATH == "0" or VIDEO_PATH == 0:
        print("📹 Using camera input...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở camera")
            return
    # elif not os.path.exists(VIDEO_PATH):
    #     print(f"⚠️ Không tìm thấy video: {VIDEO_PATH}")
    #     return
    # else:
    #     cap = cv2.VideoCapture(VIDEO_PATH)
        
    elif os.path.isdir(VIDEO_PATH):
        print(f"📁 Đang xử lý folder: {VIDEO_PATH}")

        video_files = [f for f in os.listdir(VIDEO_PATH) 
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        if not video_files:
            print("❌ Không tìm thấy video trong folder")
            return

        for video_file in video_files:
            video_path = os.path.join(VIDEO_PATH, video_file)
            print(f"\n🚀 Processing: {video_file}")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Không mở được: {video_file}")
                continue
            process_single_video(cap, video_file)

        return

    if not cap.isOpened():
        print(f"⚠️ Không thể mở video: {VIDEO_PATH}")
        return

    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default 30 nếu không có
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0  # 0 cho camera
    
    if total_frames > 0:
        print(f"📹 Video Info: {width}x{height} @ {fps}fps, Total: {total_frames} frames")
    else:
        print(f"📹 Camera Input: {width}x{height} @ {fps}fps (Live)")
    
    if logger:
        logger.info("=" * 60)
        logger.info(f"Video Processing Started")
        logger.info(f"   Video: {VIDEO_PATH}")
        logger.info(f"   Resolution: {width}x{height}")
        logger.info(f"   FPS: {fps}")
        logger.info(f"   Total Frames: {total_frames}")
        logger.info("=" * 60)
    
    # Thống kê tổng thể
    total_detections = 0
    total_tracks = 0
    frames_with_detections = 0
    frames_without_detections = 0
    class_statistics = defaultdict(int)
    all_track_ids = set()

    # Setup video writer nếu cần lưu
    out_writer = None
    if SAVE_RESULT:
        output_path = os.path.join(OUTPUT_DIR, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Đang lưu video vào: {output_path}")

    # Tạo cửa sổ
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_count = 0
    paused = False
    frame = None
    
    # Reset KCF trackers và tracking history khi bắt đầu video mới
    global kcf_trackers, kcf_boxes, tracking_history
    global kalman_filters, current_adaptive_conf, detection_quality_history
    global occluded_tracks, last_frame_results, skip_frame_counter, tracking_stability
    kcf_trackers.clear()
    kcf_boxes.clear()
    tracking_history.clear()
    kalman_filters.clear()
    occluded_tracks.clear()
    detection_quality_history.clear()
    current_adaptive_conf = CONF_THRESH
    last_frame_results = None
    skip_frame_counter = 0
    tracking_stability.clear()
    
    # FPS tracking - cải thiện để hiển thị ngay từ đầu
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    frame_times = []  # Lưu thời gian xử lý các frame gần đây
    video_start_time = time.time()  # Thời gian bắt đầu xử lý video

    print("\n🎮 Điều khiển:")
    print("   SPACE: Tạm dừng/Tiếp tục")
    print("   ESC: Thoát")
    print("   Q: Thoát\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ Đã xử lý xong video!")
                if logger:
                    logger.info("=" * 60)
                    logger.info("Video Processing Completed")
                break

            frame_count += 1
            
            # Smart Frame Skipping - Hạn chế để tăng accuracy
            should_skip = False
            if USE_SMART_SKIP and skip_frame_counter < MAX_SKIP_FRAMES and last_frame_results:
                prev_stats = last_frame_results.get('stats', {})
                prev_track_count = prev_stats.get('track_count', 0)
                prev_detection_count = prev_stats.get('detection_count', 0)
                
                # Skip chỉ khi:
                # 1. Tracking rất ổn định (nhiều tracks và ổn định trong nhiều frames)
                # 2. Và có detections (không skip khi không có detections)
                if prev_track_count >= 3 and prev_detection_count > 0:  # Tăng từ 2 lên 3, và phải có detections
                    # Kiểm tra stability - chỉ skip nếu tracking rất ổn định
                    stability_score = prev_track_count / max(1, prev_detection_count)
                    if stability_score >= SKIP_THRESHOLD_STABLE:  # Phải rất ổn định
                        should_skip = True
                        skip_frame_counter += 1
                        frame = last_frame_results.get('frame', frame).copy()
                        stats = prev_stats.copy()
                # Tắt skip khi không có detections để detect tốt hơn
                # elif SKIP_ON_NO_DETECTIONS and prev_detection_count == 0:
                #     # Không skip khi không có detections
                #     pass
            
            if not should_skip:
                skip_frame_counter = 0
                # Tính FPS - cải thiện để hiển thị ngay từ đầu
                frame_start_time = time.time()
                
                # Xử lý frame
                frame, stats = process_frame(frame, frame_count, total_frames, current_fps)
                
                # Lưu results cho frame skipping
                if USE_SMART_SKIP:
                    last_frame_results = {'frame': frame.copy(), 'stats': stats.copy()}
            else:
                # Đã skip, chỉ cần update frame count
                frame_start_time = time.time()
            
            # Cập nhật thống kê
            if stats['detection_count'] > 0:
                frames_with_detections += 1
                total_detections += stats['detection_count']
                for cls, count in stats['class_counts'].items():
                    class_statistics[cls] += count
            else:
                frames_without_detections += 1
            
            if stats['track_count'] > 0:
                total_tracks += stats['track_count']
                all_track_ids.update(stats['track_ids'])
            
            # Tính FPS sau khi xử lý
            if SHOW_FPS:
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                # Giữ tối đa 30 frame times để tính FPS trung bình
                if len(frame_times) > 30:
                    frame_times.pop(0)
                # Tính FPS từ trung bình của các frame times
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Cập nhật thống kê (lấy từ process_frame nếu có)
            # Note: Cần modify process_frame để return stats hoặc dùng global

            if out_writer:
                out_writer.write(frame)

        # Hiển thị frame (chỉ khi có frame)
        if frame is not None:
            cv2.imshow(WINDOW_NAME, frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == 27 or key == ord('q'):  # ESC hoặc Q
            print("👋 Thoát chương trình.")
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print("⏸️ Tạm dừng" if paused else "▶️ Tiếp tục")

    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    
    # Thống kê cuối cùng
    video_end_time = time.time()
    total_processing_time = video_end_time - video_start_time
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    
    # Performance summary
    perf_summary = get_perf_summary()
    
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ KẾT QUẢ")
    print("=" * 60)
    print(f"   Tổng số frames đã xử lý: {frame_count}")
    print(f"   Thời gian xử lý: {total_processing_time:.2f}s")
    print(f"   FPS trung bình: {avg_fps:.2f}")
    if frame_times:
        min_frame_time = max(frame_times)
        max_frame_time = min([t for t in frame_times if t > 0])  # Lọc bỏ 0
        if min_frame_time > 0:
            print(f"   FPS min: {1.0/min_frame_time:.2f}")
        if max_frame_time > 0:
            print(f"   FPS max: {1.0/max_frame_time:.2f}")
    
    if perf_summary:
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"   Inference time (avg): {perf_summary['avg_inference_ms']:.2f} ms")
        print(f"   Total time (avg): {perf_summary['avg_total_ms']:.2f} ms")
        print(f"   Inference time (min): {perf_summary['min_inference_ms']:.2f} ms")
        print(f"   Inference time (max): {perf_summary['max_inference_ms']:.2f} ms")
        print(f"   Estimated FPS: {perf_summary['fps']:.2f}")
    
    print("=" * 60)
    
    if logger:
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"   Total Frames Processed: {frame_count}")
        logger.info(f"   Processing Time: {total_processing_time:.2f}s")
        logger.info(f"   Average FPS: {avg_fps:.2f}")
        if frame_times:
            min_frame_time = max(frame_times)
            max_frame_time = min([t for t in frame_times if t > 0])  # Lọc bỏ 0
            if min_frame_time > 0:
                logger.info(f"   Min FPS: {1.0/min_frame_time:.2f}")
            if max_frame_time > 0:
                logger.info(f"   Max FPS: {1.0/max_frame_time:.2f}")
            avg_frame_time = sum(frame_times) / len(frame_times)
            if avg_frame_time > 0:
                logger.info(f"   Avg Frame Time: {avg_frame_time*1000:.2f}ms")
        if LOG_DETECTION_STATS:
            logger.info(f"   Frames with Detections: {frames_with_detections}")
            logger.info(f"   Frames without Detections: {frames_without_detections}")
            if class_statistics:
                logger.info("   Class Statistics:")
                for cls, count in sorted(class_statistics.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"      {cls}: {count}")
        if LOG_TRACKING_STATS and ENABLE_TRACKING:
            logger.info(f"   Total Unique Tracks: {len(all_track_ids)}")
        logger.info("=" * 60)
        logger.info("Processing Complete!")
        if LOG_TO_FILE:
            logger.info(f"Log saved to: {log_file}")

# =====================================================
# CHẠY CHƯƠNG TRÌNH
# =====================================================
if __name__ == "__main__":
    process_video()
