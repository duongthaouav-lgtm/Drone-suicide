"""
Script để tải model drone-detector từ Hugging Face
"""
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

# Tạo thư mục weights nếu chưa có
weights_dir = Path("./weights")
weights_dir.mkdir(exist_ok=True)

print("📥 Đang tải model drone-detector từ Hugging Face...")
print("   Model: marie-kjelberg/drone-detector")
print("   File: yolo11n_drone.pt\n")

try:
    # Tải model file
    model_path = hf_hub_download(
        repo_id="marie-kjelberg/drone-detector",
        filename="yolo11n_drone.pt",
        local_dir=str(weights_dir),
        local_dir_use_symlinks=False
    )
    
    print(f"✅ Tải thành công!")
    print(f"   Đường dẫn: {model_path}")
    print(f"\n💡 Để sử dụng, cập nhật MODEL_PATH trong test_models.py:")
    print(f'   MODEL_PATH = "{model_path}"')
    print(f"   hoặc")
    print(f'   MODEL_PATH = "./weights/yolo11n_drone.pt"')
    
except Exception as e:
    print(f"❌ Lỗi khi tải model: {e}")
    print("\n🔧 Giải pháp:")
    print("   1. Cài đặt huggingface_hub: pip install huggingface_hub")
    print("   2. Hoặc tải thủ công từ: https://huggingface.co/marie-kjelberg/drone-detector")
    print("   3. Lưu file yolo11n_drone.pt vào thư mục ./weights/")

