import os
import sys
import cv2
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm

# Try to import torch with error handling for Windows DLL loading issues
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, KeyboardInterrupt, Exception) as e:
    print("=" * 60)
    print("⚠️ ERROR: Failed to import PyTorch")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nThis is often caused by:")
    print("  1. Corrupted PyTorch installation")
    print("  2. Missing or incompatible CUDA libraries")
    print("  3. Antivirus blocking DLL loading")
    print("  4. Windows Store Python compatibility issues")
    print("\nSolutions:")
    print("  1. Reinstall PyTorch:")
    print("     pip uninstall torch torchvision torchaudio")
    print("     pip install torch torchvision torchaudio")
    print("  2. If using CUDA, install CPU-only version first:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("  3. Try running as Administrator")
    print("  4. Temporarily disable antivirus")
    print("  5. Use a different Python installation (not Windows Store Python)")
    print("=" * 60)
    sys.exit(1)

# Try to import ultralytics (which also imports torch)
try:
    from ultralytics import YOLO
except (ImportError, KeyboardInterrupt, Exception) as e:
    print("=" * 60)
    print("⚠️ ERROR: Failed to import Ultralytics")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nThis may be caused by PyTorch import issues.")
    print("Please fix PyTorch installation first (see above).")
    print("=" * 60)
    sys.exit(1)

# Kalman Filter (optional - cần cài: pip install filterpy)
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("Warning: filterpy not installed. Kalman Filter disabled. Install with: pip install filterpy")

# =====================================================
# CONFIG
# =====================================================
# Model paths - Có thể thử các model sau:
# 1. Model hiện tại: "./weights/best.pt"
# 2. Model mới (đã train trên dataset lớn): "./weights/yolo11n_drone.pt"
#    Tải từ: https://huggingface.co/marie-kjelberg/drone-detector
#    Chạy: python download_model.py
# MODEL_PATH = "./weights/best.pt"           # model segmentation
MODEL_PATH = "./weights/yolo11n_drone.pt"    # Uncomment để dùng model mới

VIDEO_PATH = "./VIDEO.MOV"            # đường dẫn video
OUTPUT_DIR = "./results"              # thư mục lưu kết quả

# Input Source Configuration
USE_CAMERA_INPUT = True               # True = dùng camera live, False = dùng VIDEO_PATH
CAMERA_SOURCE = 0                     # 0 = webcam mặc định, 1 = DroidCam (thường), hoặc URL/đường dẫn chuỗi
CAMERA_WIDTH = 1280                   # Ép độ phân giải đầu vào (0 = giữ mặc định)
CAMERA_HEIGHT = 720                   # Ép độ phân giải đầu vào (0 = giữ mặc định)
CAMERA_FPS = 30                       # FPS mục tiêu khi camera không trả về FPS
CAMERA_BACKEND = "dshow"              # dshow | msmf | any (Windows nên dùng dshow cho DroidCam)


def resolve_camera_backend(backend_name):
    """Map tên backend thân thiện sang flag của OpenCV."""
    if backend_name is None:
        return cv2.CAP_ANY
    backend = backend_name.lower().strip()
    mapping = {
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "directshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "mediafoundation": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "any": cv2.CAP_ANY
    }
    return mapping.get(backend, cv2.CAP_ANY)
# =====================================================
# CONFIG - TỐI ƯU HÓA
# =====================================================
# Confidence threshold - TĂNG CAO để giảm false positives
# Khuyến nghị: 0.25-0.35 cho model mới (yolo11n_drone.pt), 0.05-0.15 cho model cũ
CONF_THRESH = 0.30  # TĂNG từ 0.05 lên 0.30 để LOẠI BỎ FALSE POSITIVES mạnh hơn
IMGSZ = 800         # Input size (TĂNG từ 720 lên 800 để detect tốt hơn)
IOU_THRESH = 0.48  # IoU threshold cho NMS (TĂNG từ 0.45 lên 0.48 để GIẢM FALSE POSITIVES MẠNH HƠN)

# Adaptive Processing - TỐI ƯU ĐỂ TĂNG DETECTIONS
USE_ADAPTIVE_IMGSZ = True  # Tự động điều chỉnh image size
ADAPTIVE_IMGSZ_MIN = 640  # Size tối thiểu (nhanh)
ADAPTIVE_IMGSZ_MAX = 960  # Size tối đa (TĂNG từ 832 lên 960 để detect tốt hơn khi cần)
ADAPTIVE_IMGSZ_BASE = 800  # Size mặc định (TĂNG từ 720 lên 800 để detect tốt hơn)
DEBUG_DETECTIONS = False  # Tắt debug để giảm noise
SAVE_RESULT = True
WINDOW_NAME = "🚀 YOLO Segmentation Viewer"

# FPS Optimization
USE_HALF = True  # FP16 để tăng tốc (cần GPU)
USE_TENSORRT = False  # TensorRT optimization (cần export model trước)
BATCH_SIZE = 1  # Batch processing (tăng nếu GPU memory đủ)
ASYNC_INFERENCE = False  # Async processing (tăng FPS nhưng phức tạp hơn)

# Detection Optimization
MULTI_SCALE = False  # Multi-scale detection (chính xác hơn nhưng chậm)
TEST_TIME_AUG = False  # Test Time Augmentation (chính xác hơn nhưng rất chậm)
# DETECT_EVERY_N_FRAMES = 1  # Đã tắt vì tracking cần detection mỗi frame

# Tracking Optimization
SHOW_FPS = True
ENABLE_TRACKING = True
TRACKER_TYPE = "bytetrack"  # "bytetrack", "botsort", "kcf", "hybrid" - Hybrid: ByteTrack + KCF
TRACK_BUFFER = 50  # Tăng buffer để tracking tốt hơn
TRACK_CONF_THRESH = 0.30  # Confidence threshold riêng cho tracking (TĂNG từ 0.05 lên 0.30 để LOẠI BỎ FALSE POSITIVES)
TRACK_IOU_THRESH = 0.55  # IoU threshold cho tracking (TĂNG từ 0.50 lên 0.55 để GIẢM FALSE POSITIVES)
MIN_TRACK_FRAMES = 3  # Số frame tối thiểu để giữ track
MAX_AGE = 30  # Số frame tối đa để giữ track khi mất detection

# Temporal Smoothing - TỐI ƯU ĐỂ TĂNG FPS
USE_TEMPORAL_SMOOTHING = True  # Làm mượt bounding boxes
SMOOTHING_ALPHA = 0.8  # Trọng số cho smoothing (TĂNG từ 0.7 lên 0.8 để giảm tính toán)
SMOOTHING_HISTORY = 3  # Số frame để lưu history (GIẢM từ 5 xuống 3 để tăng FPS)

# Logging Configuration
ENABLE_LOGGING = True  # Bật logging
LOG_TO_FILE = True  # Lưu log vào file
LOG_LEVEL = logging.INFO  # DEBUG, INFO, WARNING, ERROR
LOG_DETECTION_STATS = True  # Log thống kê detections
LOG_TRACKING_STATS = True  # Log thống kê tracking
LOG_EVERY_N_FRAMES = 30  # Log mỗi N frames (0 = tắt)

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
# LOAD MODEL
# =====================================================
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    USE_HALF = False  # FP16 chỉ hoạt động trên GPU
    print(f"⚠️ Đang sử dụng CPU (chậm hơn GPU)")
    if logger:
        logger.warning("Using CPU (slower than GPU)")
elif device == 'cuda':
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

# Adaptive Confidence Threshold - LOẠI BỎ FALSE POSITIVES 0.03-0.04
USE_ADAPTIVE_CONF = True  # Tự động điều chỉnh confidence
ADAPTIVE_CONF_MIN = 0.25  # Confidence tối thiểu (TĂNG từ 0.05 lên 0.25 để LOẠI BỎ FALSE POSITIVES mạnh hơn)
ADAPTIVE_CONF_MAX = 0.50  # Confidence tối đa (TĂNG từ 0.18 lên 0.50 để có phạm vi điều chỉnh tốt hơn)
ADAPTIVE_CONF_STEP = 0.01  # Bước điều chỉnh (GIẢM từ 0.015 xuống 0.01 để điều chỉnh mịn hơn)

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
MULTI_SCALE_THRESHOLD = 3  # Số detections tối đa để trigger multi-scale (TĂNG từ 2 lên 3 để trigger THƯỜNG XUYÊN HƠN)
MULTI_SCALE_FRAME_INTERVAL = 1  # Dùng multi-scale mỗi N frames (GIẢM từ 2 xuống 1 để detect tốt hơn - dùng mỗi frame khi cần)

# Smart Frame Skipping - TẮT ĐỂ TĂNG DETECTIONS
USE_SMART_SKIP = False  # TẮT frame skipping để xử lý tất cả frames (tăng detections)
SKIP_THRESHOLD_STABLE = 0.98  # Tracking stability threshold (không dùng khi tắt skip)
MAX_SKIP_FRAMES = 0  # Tắt skip hoàn toàn để detect mọi frame
SKIP_ON_NO_DETECTIONS = False  # Tắt skip khi không có detections
SKIP_CONSECUTIVE_NO_DET = 10  # Không dùng khi tắt skip

# Batch Processing - Tối ưu
USE_BATCH_PROCESSING = False  # Xử lý nhiều frames cùng lúc (tăng FPS nhưng tốn memory)
BATCH_SIZE = 2  # Số frames trong một batch (giảm để tránh OOM)

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
    
    # Điều chỉnh confidence - LOẠI BỎ FALSE POSITIVES 0.03-0.04
    # Chỉ giảm threshold khi detection rate RẤT thấp (< 50%) để tránh false positives
    if detection_rate < 0.50:  # Detection rate rất thấp -> GIẢM THRESHOLD VỪA PHẢI (TĂNG từ 60% lên 50%)
        current_adaptive_conf = max(ADAPTIVE_CONF_MIN, current_adaptive_conf - ADAPTIVE_CONF_STEP * 1.5)  # Giảm từ 2.0 xuống 1.5
    elif detection_rate < 0.55:  # Detection rate thấp -> giảm rất nhẹ
        current_adaptive_conf = max(ADAPTIVE_CONF_MIN, current_adaptive_conf - ADAPTIVE_CONF_STEP * 0.5)  # Giảm từ 1.0 xuống 0.5
    elif detection_rate < 0.70:  # Detection rate trung bình -> không giảm
        # Không giảm threshold khi detection rate >= 55% để tránh false positives
        pass
    elif avg_quality < 0.4:  # Quality rất thấp -> giảm threshold rất nhẹ (TĂNG từ 0.5 xuống 0.4)
        current_adaptive_conf = max(ADAPTIVE_CONF_MIN, current_adaptive_conf - ADAPTIVE_CONF_STEP * 0.1)  # Giảm từ 0.2 xuống 0.1
    elif avg_quality > 0.80:  # Quality cao -> tăng threshold (GIẢM từ 0.85 xuống 0.80 để tăng sớm hơn)
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
# XỬ LÝ FRAME VỚI TRACKING (TỐI ƯU) + KCF
# =====================================================
def process_frame(frame, frame_count=0, total_frames=0, fps=0):
    """Xử lý một frame từ video với tracking - tối ưu cho tốc độ và accuracy"""
    global kcf_trackers, kcf_boxes, tracking_history
    global kalman_filters, current_adaptive_conf, detection_quality_history
    global occluded_tracks, last_frame_results, skip_frame_counter, tracking_stability
    
    # Adaptive Confidence Threshold
    conf_thresh_to_use = CONF_THRESH
    if USE_ADAPTIVE_CONF:
        conf_thresh_to_use = adaptive_confidence_adjustment(detection_quality_history)
        # Đảm bảo không bao giờ thấp hơn CONF_THRESH để loại bỏ false positives
        conf_thresh_to_use = max(conf_thresh_to_use, CONF_THRESH)
        # Nếu bật tracking, cũng đảm bảo không thấp hơn TRACK_CONF_THRESH
        if ENABLE_TRACKING:
            conf_thresh_to_use = max(conf_thresh_to_use, TRACK_CONF_THRESH)
    
    # Adaptive Image Size - TỐI ƯU ĐỂ TĂNG DETECTIONS
    imgsz_to_use = IMGSZ  # Dùng IMGSZ mặc định (800) để detect tốt hơn
    if USE_ADAPTIVE_IMGSZ:
        # ƯU TIÊN TĂNG DETECTIONS - phản ứng nhanh hơn
        if len(detection_quality_history) > 2:  # GIẢM từ 3 xuống 2 để phản ứng NHANH HƠN
            recent_no_detections = sum(1 for q in detection_quality_history[-6:] if q == 0)  # GIẢM từ 8 xuống 6 để phản ứng nhanh hơn
            detection_rate = 1.0 - (recent_no_detections / min(6, len(detection_quality_history)))
            
            # Nếu detection rate < 60%, tăng size để detect tốt hơn (GIẢM từ 70% xuống 60% để ít tăng size hơn)
            if detection_rate < 0.60:
                imgsz_to_use = ADAPTIVE_IMGSZ_MAX  # Dùng size lớn (960) để detect tốt hơn
            elif detection_rate > 0.80:  # GIẢM từ 0.85 xuống 0.80 để giảm size sớm hơn
                imgsz_to_use = ADAPTIVE_IMGSZ_MIN  # Dùng size nhỏ (640) để tăng FPS
            else:
                imgsz_to_use = ADAPTIVE_IMGSZ_BASE  # Dùng size base (800)
        else:
            imgsz_to_use = ADAPTIVE_IMGSZ_BASE  # Mặc định dùng size base
    
    # Smart Multi-scale Detection - GIẢM FALSE POSITIVES
    use_multi_scale_now = False
    if USE_SMART_MULTI_SCALE and len(detection_quality_history) > 0:
        recent_detections = sum(1 for q in detection_quality_history[-5:] if q > 0)
        # Dùng multi-scale mỗi N frames để giảm overhead
        # Trigger khi: rất ít detections (GIẢM từ <= 3 xuống <= 1 để ít trigger hơn)
        if frame_count % MULTI_SCALE_FRAME_INTERVAL == 0 and recent_detections <= 1:
            # Nếu rất ít detections, có thể có objects nhỏ -> dùng multi-scale
            use_multi_scale_now = True
            imgsz_to_use = int(imgsz_to_use * 1.15)  # Giảm từ 1.20 xuống 1.15 để ít aggressive hơn
    
    # Sử dụng track() nếu bật tracking, nếu không dùng predict()
    if ENABLE_TRACKING and TRACKER_TYPE in ["bytetrack", "botsort", "hybrid"]:
        tracker_name = "botsort.yaml" if TRACKER_TYPE == "botsort" else "bytetrack.yaml"
        results = model.track(
            frame, 
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
            max_det=50,  # GIẢM từ 1000 xuống 50 để GIẢM FALSE POSITIVES mạnh hơn
            classes=None,  # Detect tất cả classes
            retina_masks=False,  # Tắt để tăng tốc
            stream=False
        )
    else:
        results = model.predict(
            frame, 
            conf=conf_thresh_to_use,
            iou=IOU_THRESH,
            imgsz=imgsz_to_use, 
            verbose=DEBUG_DETECTIONS,
            device=device,
            half=USE_HALF,
            show=False,
            agnostic_nms=False,
            max_det=50,  # GIẢM từ 1000 xuống 50 để GIẢM FALSE POSITIVES mạnh hơn
            classes=None,
            retina_masks=False,
            stream=False
        )
    
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
                        kcf_tracker = cv2.legacy.TrackerCSRT_create()
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
        if r.masks is not None:
            # Sử dụng polygon coordinates thay vì resize mask (nhanh hơn)
            if hasattr(r.masks, 'xy') and r.masks.xy is not None:
                for idx, mask_poly in enumerate(r.masks.xy):
                    if len(mask_poly) > 0:
                        pts = np.int32([mask_poly])
                        
                        # Màu mask dựa trên track ID nếu có (match với box index)
                        if track_ids is not None and idx < len(track_ids):
                            color_id = track_ids[idx] % 255
                            mask_color = (
                                int(255 * np.sin(color_id * 0.1) ** 2),
                                int(255 * np.sin(color_id * 0.1 + 2) ** 2),
                                int(255 * np.sin(color_id * 0.1 + 4) ** 2)
                            )
                        else:
                            mask_color = (0, 0, 255)  # Màu đỏ mặc định
                        
                        # Vẽ filled polygon với độ trong suốt
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [pts], mask_color)
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
                        # Vẽ contour
                        cv2.polylines(annotated_frame, [pts], True, mask_color, 2)
            else:
                # Fallback: xử lý mask data nếu không có polygon
                masks = r.masks.data.cpu().numpy()
                for idx, mask in enumerate(masks):
                    mask_resized = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Màu mask dựa trên track ID
                    if track_ids is not None and idx < len(track_ids):
                        color_id = track_ids[idx] % 255
                        mask_color = (
                            int(255 * np.sin(color_id * 0.1) ** 2),
                            int(255 * np.sin(color_id * 0.1 + 2) ** 2),
                            int(255 * np.sin(color_id * 0.1 + 4) ** 2)
                        )
                    else:
                        mask_color = (0, 0, 255)
                    
                    overlay = annotated_frame.copy()
                    overlay[mask_binary == 1] = mask_color
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

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
        # Nếu có raw detections nhưng không pass threshold, quality = 0.1 (TĂNG từ 0.05 để GIẢM FALSE POSITIVES MẠNH HƠN)
        if raw_detection_count > 0 and detection_count == 0:
            quality_score = 0.1  # TĂNG từ 0.05 lên 0.1 để ít aggressive hơn nhiều
        # Nếu có nhiều raw detections nhưng ít final detections, cần giảm threshold
        elif raw_detection_count > detection_count * 1.5:  # TĂNG từ 1.3 lên 1.5 để ít nhạy hơn nhiều
            quality_score = 0.3  # TĂNG từ 0.2 lên 0.3 để ít aggressive hơn nhiều
        
        detection_quality_history.append(quality_score)
        if len(detection_quality_history) > 30:  # Giảm từ 40 xuống 30 để phản ứng nhanh hơn
            detection_quality_history.pop(0)
    
    # Cleanup Kalman filters không còn active
    if USE_KALMAN_FILTER and track_ids_set:
        active_track_ids = set(track_ids_set)
        for tid in list(kalman_filters.keys()):
            if tid not in active_track_ids:
                del kalman_filters[tid]
    
    # Return stats để cập nhật thống kê
    stats = {
        'detection_count': detection_count,
        'track_count': track_count,
        'class_counts': dict(class_counts),
        'track_ids': list(track_ids_set)
    }
    
    return annotated_frame, stats

# =====================================================
# XỬ LÝ VIDEO
# =====================================================
def process_video():
    """Xử lý video file hoặc live camera input (DroidCam/USB/RTSP)."""
    using_camera = USE_CAMERA_INPUT
    cap = None
    input_label = VIDEO_PATH
    
    if using_camera:
        # Chuẩn hóa nguồn camera (hỗ trợ int, str số, URL)
        source_value = CAMERA_SOURCE
        if source_value is None:
            source_value = 0
        elif isinstance(source_value, str):
            stripped = source_value.strip()
            source_value = int(stripped) if stripped.isdigit() else stripped
        
        backend_flag = resolve_camera_backend(CAMERA_BACKEND)
        try:
            if isinstance(source_value, int):
                cap = cv2.VideoCapture(source_value, backend_flag)
                input_label = f"Camera #{source_value}"
            else:
                cap = cv2.VideoCapture(source_value)
                input_label = str(source_value)
        except Exception:
            cap = cv2.VideoCapture(source_value)
            input_label = str(source_value)
        
        if not cap or not cap.isOpened():
            print(f"⚠️ Không thể mở camera: {CAMERA_SOURCE}")
            print("👉 Kiểm tra DroidCam/ứng dụng camera và đảm bảo đã cho phép truy cập.")
            return
        
        if CAMERA_WIDTH:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        if CAMERA_HEIGHT:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if CAMERA_FPS:
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else CAMERA_FPS or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAMERA_WIDTH or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAMERA_HEIGHT or 480
        total_frames = 0  # Camera stream không biết trước tổng frame
        
        print(f"📷 Camera Info: {width}x{height} @ {fps}fps | Source: {input_label}")
        
        if logger:
            logger.info("=" * 60)
            logger.info("Camera Stream Processing Started")
            logger.info(f"   Source: {input_label}")
            logger.info(f"   Target Resolution: {width}x{height}")
            logger.info(f"   Target FPS: {fps}")
            logger.info("=" * 60)
    else:
        if not os.path.exists(VIDEO_PATH):
            print(f"⚠️ Không tìm thấy video: {VIDEO_PATH}")
            return
        
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"⚠️ Không thể mở video: {VIDEO_PATH}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Info: {width}x{height} @ {fps}fps, Total: {total_frames} frames")
        
        if logger:
            logger.info("=" * 60)
            logger.info("Video Processing Started")
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
        record_fps = fps if fps and fps > 0 else (CAMERA_FPS or 30)
        out_writer = cv2.VideoWriter(output_path, fourcc, record_fps, (width, height))
        print(f"💾 Đang lưu {'camera' if using_camera else 'video'} vào: {output_path}")

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
