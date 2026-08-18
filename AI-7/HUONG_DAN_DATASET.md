# 📚 Hướng Dẫn Sử Dụng Dataset Drone-Detection

## 📋 Tổng Quan

Dataset `pathikg/drone-detection-dataset` là dataset lớn với:
- **~54k ảnh** tổng cộng
- **51.4k ảnh train** + **2.6k ảnh test**
- Format: Hugging Face Dataset (cần chuyển đổi sang YOLO format)
- **Nguồn:** https://huggingface.co/datasets/pathikg/drone-detection-dataset

## 🎯 Tại Sao Cần Dataset Này?

1. **Model hiện tại** (`best.pt`) có quá nhiều false positives
2. **Model mới** (`yolo11n_drone.pt`) đã được train trên dataset này nhưng vẫn có thể cần fine-tune cho môi trường indoor
3. **Fine-tune với dữ liệu của bạn** sẽ giúp model học được đặc thù môi trường indoor

## 🔧 Cài Đặt

### Bước 1: Cài đặt thư viện

```bash
pip install datasets pillow tqdm
```

### Bước 2: Tải và chuẩn bị dataset

```bash
python download_and_prepare_dataset.py
```

Script này sẽ:
- Tải dataset từ Hugging Face
- Chuyển đổi sang format YOLO (images + labels)
- Tạo file `dataset.yaml` để sử dụng cho training
- Lưu vào thư mục `./datasets/drone-detection/`

**Thời gian:** ~30-60 phút tùy vào tốc độ internet

### Bước 3: Kiểm tra dataset

Sau khi tải xong, kiểm tra cấu trúc:

```
datasets/drone-detection/
├── images/
│   ├── train/     # 51.4k ảnh
│   └── val/       # 2.6k ảnh
├── labels/
│   ├── train/     # 51.4k file .txt
│   └── val/       # 2.6k file .txt
└── dataset.yaml   # Config file
```

## 🚀 Sử Dụng Dataset

### Option 1: Train model mới từ đầu

```bash
python train_drone_model.py
```

### Option 2: Fine-tune model hiện tại

Chỉnh sửa `train_drone_model.py`:

```python
BASE_MODEL = "./weights/yolo11n_drone.pt"  # Dùng model đã có
EPOCHS = 50  # Fine-tune ít epochs hơn
```

### Option 3: Kết hợp với dữ liệu của bạn

1. **Thu thập ảnh indoor:**
   - Ảnh có drone (gán nhãn)
   - Ảnh KHÔNG có drone (negative samples - quan trọng!)

2. **Gán nhãn bằng công cụ:**
   - [LabelImg](https://github.com/tzutalin/labelImg)
   - [Roboflow](https://roboflow.com/)
   - [CVAT](https://cvat.org/)

3. **Tạo dataset.yaml mới:**

```yaml
path: ./datasets/combined
train: images/train
val: images/val

nc: 1
names:
  0: drone
```

4. **Train với dataset kết hợp:**

```python
model = YOLO("yolo11n_drone.pt")
model.train(
    data="./datasets/combined/dataset.yaml",
    epochs=100,
    imgsz=640
)
```

## 📊 Cấu Trúc Dataset YOLO

Mỗi ảnh có file label tương ứng:

**Ảnh:** `000001.jpg`  
**Label:** `000001.txt`

Format label (YOLO):
```
class_id x_center y_center width height
```

Ví dụ:
```
0 0.5 0.5 0.2 0.3
```
- `0` = class drone
- `0.5 0.5` = center của bbox (normalized 0-1)
- `0.2 0.3` = width và height (normalized 0-1)

## ⚙️ Tối Ưu Training

### GPU Memory

Nếu GPU memory không đủ:
```python
BATCH_SIZE = 8  # Giảm từ 16 xuống 8
IMGSZ = 640     # Giảm từ 800 xuống 640
```

### Tốc Độ Training

- **GPU tốt (RTX 3090, A100):** ~2-4 giờ cho 100 epochs
- **GPU trung bình (RTX 3060):** ~6-10 giờ
- **CPU:** Không khuyến nghị (quá chậm)

### Early Stopping

YOLO tự động:
- Lưu `best.pt` (model tốt nhất trên validation)
- Lưu `last.pt` (model cuối cùng)
- Tạo plots và metrics

## 🎯 Kết Quả Mong Đợi

Sau khi train, bạn sẽ có:
- **best.pt:** Model tốt nhất (dùng cho inference)
- **last.pt:** Model cuối cùng
- **Metrics:** Precision, Recall, mAP
- **Plots:** Loss curves, confusion matrix

## 📝 Lưu Ý Quan Trọng

1. **Negative Samples:** Dataset này có nhiều ảnh không có drone, giúp model học được background
2. **Domain Gap:** Dataset chủ yếu là outdoor, bạn cần thêm indoor data
3. **Validation:** Luôn kiểm tra model trên dữ liệu thực tế của bạn
4. **Confidence Threshold:** Sau khi train, điều chỉnh threshold phù hợp

## 🔗 Tài Liệu Tham Khảo

- [YOLO Training Docs](https://docs.ultralytics.com/modes/train/)
- [Dataset Format](https://docs.ultralytics.com/datasets/)
- [Hugging Face Dataset](https://huggingface.co/datasets/pathikg/drone-detection-dataset)

