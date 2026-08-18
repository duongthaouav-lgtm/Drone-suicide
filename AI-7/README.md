# 🚁 AI-UAV-DRONE Detection System

Hệ thống phát hiện và theo dõi drone sử dụng YOLO với khả năng giảm false positives.

## 📋 Tính Năng

- ✅ **Real-time Detection**: Phát hiện drone trong thời gian thực từ camera hoặc video
- ✅ **Multi-Object Tracking**: Theo dõi nhiều drone cùng lúc với ByteTrack + KCF (Hybrid)
- ✅ **Segmentation**: Hỗ trợ instance segmentation
- ✅ **False Positive Reduction**: Tối ưu confidence threshold và filtering
- ✅ **Adaptive Processing**: Tự động điều chỉnh image size và confidence
- ✅ **GPU Acceleration**: Hỗ trợ CUDA với FP16

## 🚀 Cài Đặt

### Yêu Cầu

- Python 3.8+
- CUDA-capable GPU (khuyến nghị) hoặc CPU
- Webcam hoặc video file

### Cài Đặt Dependencies

```bash
pip install ultralytics opencv-python numpy torch torchvision
pip install filterpy  # Cho Kalman Filter (optional)
pip install huggingface_hub datasets pillow tqdm  # Cho dataset tools
```

## 📦 Models

### Model Hiện Tại

- `weights/best.pt` - Model gốc (có thể có nhiều false positives)
- `weights/yolo11n_drone.pt` - Model đã được train trên dataset lớn (~54k ảnh)

### Tải Model Mới

```bash
python download_model.py
```

Model sẽ được tải từ: https://huggingface.co/marie-kjelberg/drone-detector

## 🎮 Sử Dụng

### Chạy Detection với Camera

```bash
python test_models.py
```

Cấu hình trong `test_models.py`:
- `USE_CAMERA_INPUT = True` - Dùng camera live
- `CAMERA_SOURCE = 0` - Webcam mặc định
- `CONF_THRESH = 0.30` - Confidence threshold (đã tối ưu)

### Chạy với Video File

```python
USE_CAMERA_INPUT = False
VIDEO_PATH = "./VIDEO.MOV"
```

### Điều Khiển

- **SPACE**: Tạm dừng/Tiếp tục
- **ESC/Q**: Thoát

## 📊 Dataset và Training

### Tải Dataset

Dataset lớn (~54k ảnh) từ Hugging Face:

```bash
python download_and_prepare_dataset.py
```

Dataset: https://huggingface.co/datasets/pathikg/drone-detection-dataset

### Train/Fine-tune Model

```bash
python train_drone_model.py
```

Xem chi tiết trong `HUONG_DAN_DATASET.md`

## ⚙️ Cấu Hình

### Confidence Threshold

Đã được tối ưu để giảm false positives:

```python
CONF_THRESH = 0.30  # Tăng từ 0.05 để loại bỏ false positives
TRACK_CONF_THRESH = 0.30
ADAPTIVE_CONF_MIN = 0.25
ADAPTIVE_CONF_MAX = 0.50
```

### Tracking

- **Hybrid Mode**: ByteTrack + KCF
- **Kalman Filter**: Dự đoán chuyển động
- **Temporal Smoothing**: Làm mượt bounding boxes

## 📁 Cấu Trúc Project

```
AI-7/
├── test_models.py              # Main detection script
├── download_model.py            # Tải model từ Hugging Face
├── download_and_prepare_dataset.py  # Tải và chuẩn bị dataset
├── train_drone_model.py         # Script training
├── weights/                     # Model weights (không push lên Git)
├── results/                     # Kết quả detection (logs, videos)
├── datasets/                    # Dataset (không push lên Git)
├── README.md                    # File này
└── HUONG_DAN_DATASET.md        # Hướng dẫn chi tiết về dataset
```

## 🔧 Troubleshooting

### False Positives Quá Nhiều

1. **Tăng confidence threshold:**
   ```python
   CONF_THRESH = 0.40  # hoặc 0.50
   ```

2. **Sử dụng model mới:**
   ```python
   MODEL_PATH = "./weights/yolo11n_drone.pt"
   ```

3. **Fine-tune với dữ liệu của bạn:**
   - Thu thập ảnh indoor
   - Gán nhãn (có và không có drone)
   - Train lại model

### GPU Không Hoạt Động

- Kiểm tra CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Cài đặt PyTorch với CUDA: https://pytorch.org/get-started/locally/

### Camera Không Mở Được

- Kiểm tra `CAMERA_SOURCE` (0, 1, hoặc URL)
- Thử `CAMERA_BACKEND = "dshow"` (Windows) hoặc `"msmf"`

## 📚 Tài Liệu Tham Khảo

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Model trên Hugging Face](https://huggingface.co/marie-kjelberg/drone-detector)
- [Dataset trên Hugging Face](https://huggingface.co/datasets/pathikg/drone-detection-dataset)

## 📝 License

MIT License

## 👤 Author

dkzdragon02

## 🙏 Acknowledgments

- Model được train dựa trên dataset từ [pathikg/drone-detection-dataset](https://huggingface.co/datasets/pathikg/drone-detection-dataset)
- Model pretrained từ [marie-kjelberg/drone-detector](https://huggingface.co/marie-kjelberg/drone-detector)
- Sử dụng [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

