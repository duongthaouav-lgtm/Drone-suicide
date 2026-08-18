"""
Script để train/fine-tune model drone detection
Sử dụng dataset đã tải về hoặc dataset hiện có
"""
from ultralytics import YOLO
from pathlib import Path

# =====================================================
# CONFIG TRAINING
# =====================================================

# Model để fine-tune (có thể dùng pretrained hoặc model hiện tại)
BASE_MODEL = "yolo11n.pt"  # Hoặc "./weights/yolo11n_drone.pt" để tiếp tục train
# BASE_MODEL = "./weights/yolo11n_drone.pt"  # Uncomment để tiếp tục train từ model đã có

# Dataset config
DATASET_YAML = "./datasets/drone-detection/dataset.yaml"  # Nếu đã tải dataset
# DATASET_YAML = "path/to/your/dataset.yaml"  # Hoặc dataset của bạn

# Training parameters
EPOCHS = 100
IMGSZ = 640  # 640x640 hoặc 800x800
BATCH_SIZE = 16  # Giảm nếu GPU memory không đủ
LEARNING_RATE = 0.001
DEVICE = 0  # 0 = GPU đầu tiên, 'cpu' = CPU

# Data augmentation (tự động trong YOLO)
AUGMENT = True

# Validation
VAL_SPLIT = 0.2  # 20% dữ liệu dùng cho validation

print("=" * 60)
print("🚀 TRAIN/FINE-TUNE DRONE DETECTION MODEL")
print("=" * 60)
print(f"Base Model: {BASE_MODEL}")
print(f"Dataset: {DATASET_YAML}")
print(f"Epochs: {EPOCHS}")
print(f"Image Size: {IMGSZ}")
print(f"Batch Size: {BATCH_SIZE}")
print("=" * 60)

# Kiểm tra dataset
if not Path(DATASET_YAML).exists():
    print(f"\n❌ Không tìm thấy dataset: {DATASET_YAML}")
    print("💡 Chạy script download_and_prepare_dataset.py trước!")
    print("   hoặc tạo dataset.yaml của riêng bạn")
    exit(1)

# Load model
print(f"\n📦 Đang load model: {BASE_MODEL}")
try:
    model = YOLO(BASE_MODEL)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    exit(1)

# Train model
print(f"\n🎯 Bắt đầu training...")
print("   (Quá trình này có thể mất vài giờ tùy vào GPU và số epochs)")

try:
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        device=DEVICE,
        augment=AUGMENT,
        val=True,
        plots=True,  # Tạo plots
        save=True,   # Lưu checkpoints
        project="./runs/detect",  # Thư mục lưu kết quả
        name="drone_detection",  # Tên experiment
        exist_ok=True,  # Overwrite nếu đã tồn tại
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN THÀNH!")
    print("=" * 60)
    print(f"📁 Kết quả được lưu tại: {results.save_dir}")
    print(f"📊 Best model: {results.save_dir}/weights/best.pt")
    print(f"📊 Last model: {results.save_dir}/weights/last.pt")
    print("\n💡 Để sử dụng model mới:")
    print(f'   MODEL_PATH = "{results.save_dir}/weights/best.pt"')
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Lỗi khi training: {e}")
    import traceback
    traceback.print_exc()

