"""
Script để tải và chuẩn bị dataset drone-detection từ Hugging Face
Chuyển đổi sang format YOLO để sử dụng cho training
"""
import os
from pathlib import Path
from datasets import load_dataset
import shutil
from tqdm import tqdm
import json

# Cấu hình
DATASET_NAME = "pathikg/drone-detection-dataset"
OUTPUT_DIR = Path("./datasets/drone-detection")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
TRAIN_IMAGES = IMAGES_DIR / "train"
TRAIN_LABELS = LABELS_DIR / "train"
VAL_IMAGES = IMAGES_DIR / "val"
VAL_LABELS = LABELS_DIR / "val"

# Tạo thư mục
for dir_path in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("📥 TẢI VÀ CHUẨN BỊ DATASET DRONE-DETECTION")
print("=" * 60)
print(f"Dataset: {DATASET_NAME}")
print(f"Output: {OUTPUT_DIR}\n")

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Chuyển đổi bbox từ format [x, y, width, height] sang YOLO format [x_center, y_center, width, height] (normalized)
    """
    x, y, w, h = bbox
    
    # Tính center và normalize
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    
    return [x_center, y_center, width_norm, height_norm]

def process_dataset(split_name, output_images_dir, output_labels_dir):
    """Xử lý một split của dataset"""
    print(f"\n📦 Đang tải {split_name} split...")
    
    try:
        dataset = load_dataset(DATASET_NAME, split=split_name)
        print(f"   Tổng số ảnh: {len(dataset)}")
        
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            # Lấy thông tin
            image = item['image']
            image_id = item['image_id']
            width = item['width']
            height = item['height']
            objects = item['objects']
            
            # Lưu ảnh
            image_filename = f"{image_id:06d}.jpg"
            image_path = output_images_dir / image_filename
            image.save(image_path)
            
            # Chuyển đổi và lưu labels
            label_filename = f"{image_id:06d}.txt"
            label_path = output_labels_dir / label_filename
            
            bboxes = objects.get('bbox', [])
            categories = objects.get('category', [])
            
            with open(label_path, 'w') as f:
                for bbox, category in zip(bboxes, categories):
                    # Chuyển đổi bbox sang YOLO format
                    yolo_bbox = convert_bbox_to_yolo(bbox, width, height)
                    # YOLO format: class_id x_center y_center width height
                    # Category 0 = drone
                    f.write(f"{category} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
        
        print(f"✅ Đã xử lý {len(dataset)} ảnh trong {split_name} split")
        return len(dataset)
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {split_name}: {e}")
        return 0

# Xử lý train và test splits
print("\n🚀 Bắt đầu tải và xử lý dataset...")
train_count = process_dataset("train", TRAIN_IMAGES, TRAIN_LABELS)
val_count = process_dataset("test", VAL_IMAGES, VAL_LABELS)  # test split dùng làm validation

# Tạo file dataset.yaml cho YOLO
yaml_content = f"""# Drone Detection Dataset
# Dataset từ: {DATASET_NAME}
# Tổng số ảnh: Train={train_count}, Val={val_count}

path: {OUTPUT_DIR.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names:
  0: drone  # class names
"""

yaml_path = OUTPUT_DIR / "dataset.yaml"
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print("\n" + "=" * 60)
print("✅ HOÀN THÀNH!")
print("=" * 60)
print(f"📁 Dataset đã được lưu tại: {OUTPUT_DIR}")
print(f"   - Train images: {train_count} ảnh")
print(f"   - Val images: {val_count} ảnh")
print(f"   - Config file: {yaml_path}")
print("\n💡 Để train model, sử dụng:")
print(f"   from ultralytics import YOLO")
print(f"   model = YOLO('yolo11n.pt')  # hoặc model hiện tại")
print(f"   model.train(data='{yaml_path}', epochs=100, imgsz=640)")
print("=" * 60)

