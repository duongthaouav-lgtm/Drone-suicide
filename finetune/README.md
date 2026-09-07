# 🧠 Module Finetune: Huấn Luyện & Tối Ưu Mô Hình Nhận Diện Drone

Thư mục `finetune/` chứa toàn bộ quy trình tiền xử lý dữ liệu, tinh chỉnh siêu tham số (Hyperparameter Tuning), huấn luyện mô hình YOLOv8n chuyên biệt cho mục tiêu Drone trên không, và xuất trọng số phục vụ triển khai nhúng.

---

## 📑 Danh Sách Tệp Tin

| Tệp tin | Vai trò chính |
| :--- | :--- |
| [**`data.yaml`**](file:///d:/AI/finetune/data.yaml) | Tệp cấu hình đường dẫn dataset và định nghĩa lớp đối tượng (`['Drone']`). |
| [**`train.py`**](file:///d:/AI/finetune/train.py) | Huấn luyện mô hình YOLOv8n với bộ siêu tham số đã được tối ưu hóa. |
| [**`tune.py`**](file:///d:/AI/finetune/tune.py) | Tự động tìm kiếm siêu tham số bằng thuật toán di truyền (Genetic Evolution). |
| [**`split_dataset.py`**](file:///d:/AI/finetune/split_dataset.py) | Trích xuất và phân chia ảnh nền (Background/Negative images) vào các tập train/val/test. |
| [**`create_pic.py`**](file:///d:/AI/finetune/create_pic.py) | Tự động tạo các tệp nhãn rỗng `.txt` cho ảnh nền nhằm chống báo động giả (False Positive). |
| [**`test.py`**](file:///d:/AI/finetune/test.py) | Chạy kiểm thử suy luận (Inference Test) trên tập ảnh kiểm tra với mô hình `best.pt`. |
| [**`train.ipynb`**](file:///d:/AI/finetune/train.ipynb) | Jupyter Notebook hỗ trợ huấn luyện tương tác trên Google Colab / Kaggle. |
| [**`yolov8n.pt`**](file:///d:/AI/finetune/yolov8n.pt) | Trọng số gốc YOLOv8 nano từ Ultralytics. |
| [**`best.pt`**](file:///d:/AI/finetune/best.pt) | Trọng số tối ưu nhất sau khi hoàn thành huấn luyện. |

---

## 📊 Cấu Trúc Dataset & Xử Lý Dữ Liệu Nền (Negative Samples)

Dataset được kế thừa từ Roboflow Universe (`lngs-workspace/drone-detection-yhkcr-urblg/2`):
*   **Số lượng lớp**: 1 lớp (`nc: 1`, tên: `Drone`).
*   **Định dạng nhãn**: Chuẩn YOLO (`<class_id> <x_center> <y_center> <width> <height>`).

### Kỹ thuật chống báo động giả (False Positives):
Trong tác chiến UAV thực tế, bầu trời có mây, chim, lá cây hoặc các cấu trúc nhà cửa rất dễ gây nhận diện sai. Hai script [**`create_pic.py`**](file:///d:/AI/finetune/create_pic.py) và [**`split_dataset.py`**](file:///d:/AI/finetune/split_dataset.py) thực hiện:
1. Quét toàn bộ ảnh phong cảnh/nền không chứa drone.
2. Sinh file nhãn `.txt` hoàn toàn rỗng (0 bytes).
3. Đưa vào tập huấn luyện theo tỉ lệ (200 train, 60 val, 40 test). Nhờ đó mô hình học được thuộc tính của nền và giảm triệt để tình trạng bắt nhầm vật thể lạ.

---

## ⚙️ Huấn Luyện Mô Hình (`train.py`)

Cấu hình huấn luyện trong [**`train.py`**](file:///d:/AI/finetune/train.py) được điều chỉnh dựa trên kết quả tối ưu từ quá trình `tune.py`:

```python
model.train(
    data="data.yaml",
    epochs=110,           # Số chu kỳ huấn luyện
    imgsz=512,            # Kích thước ảnh chuẩn hóa
    batch=64,             # Kích thước batch
    workers=2,
    lr0=0.001,            # Tốc độ học ban đầu
    lrf=0.01318,          # Tỉ lệ học cuối cùng
    momentum=0.940,       # Momentum tối ưu
    weight_decay=0.0005,  # Chống quá khớp (overfitting)

    # Hệ số hàm mất mát (Loss Gains)
    box=8.46,             # Hệ số phạt bounding box (ưu tiên độ chính xác khung bắt)
    cls=0.5,              # Phạt phân loại lớp
    dfl=1.06,             # Distribution Focal Loss

    # Tăng cường dữ liệu (Data Augmentation)
    hsv_h=0.0015,
    hsv_s=0.807,
    hsv_v=0.361,
    scale=0.5,            # Co giãn tỉ lệ đối tượng
    fliplr=0.5,           # Lật ngang ảnh
    mosaic=0.9,           # Ghép 4 ảnh tăng cường bối cảnh
    close_mosaic=10,      # Tắt mosaic trong 10 epoch cuối để ổn định hội tụ
    project="runs",
    name="drone_detector_train"
)
```

---

## 🧬 Tinh Chỉnh Siêu Tham Số (`tune.py`)

Script [**`tune.py`**](file:///d:/AI/finetune/tune.py) sử dụng giải thuật tiến hóa (Genetic Algorithm):
*   Chạy 15 thế hệ (`iterations = 15`), mỗi thế hệ 35 epochs.
*   Cố định các tham số cơ bản và để thuật toán tự đột biến các tham số tăng cường (HSV, Scaling, Translation, v.v.).
*   Kết quả tốt nhất được tự động ghi nhận tại thư mục `runs/drone_detector_tune/best_hyperparameters.yaml`.

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chuẩn bị dữ liệu nền (Negative Samples)
```bash
# Tạo file nhãn rỗng cho ảnh chụp môi trường thực tế
python create_pic.py

# Phân bổ ảnh nền vào tập dữ liệu chính
python split_dataset.py
```

### 2. Huấn luyện mô hình
```bash
python train.py
```

### 3. Đánh giá mô hình đã huấn luyện
```bash
python test.py
```
Kết quả ảnh nhận diện sau khi test sẽ được lưu tại `runs/detect/predict/`.

### 4. Xuất trọng số & Lượng tử hóa (Quantization FP8 / INT8 / OpenVINO / ONNX)
Để chuẩn bị nạp vào module `AI-6/YOLOv8/` chạy trên máy tính nhúng Raspberry Pi 5 hoặc các bộ gia tốc AI:
```bash
# Xuất sang OpenVINO (hỗ trợ FP16 / INT8 / FP8)
yolo export model=runs/drone_detector_train/weights/best.pt format=openvino imgsz=512

# Xuất sang ONNX (chuẩn bị cho lượng tử hóa FP8 / INT8 hoặc nạp vào Hailo NPU)
yolo export model=runs/drone_detector_train/weights/best.pt format=onnx imgsz=512 opset=12
```

> [!TIP]
> **Lượng tử hóa FP8 (Floating Point 8-bit)**:
> Dự án áp dụng kỹ thuật lượng tử hóa trọng số về **FP8** (E4M3 / E5M2). So với INT8 truyền thống, FP8 duy trì dải động (Dynamic Range) vượt trội giúp hạn chế suy giảm độ chính xác khi phát hiện các drone mục tiêu kích thước nhỏ ở khoảng cách xa, đồng thời giảm 50% dung lượng mô hình và tối ưu hóa băng thông bộ nhớ trên phần cứng tăng tốc.

Sau khi xuất và lượng tử hóa xong, sao chép thư mục mô hình (ví dụ: `best_openvino_model/` hoặc mô hình đã lượng tử hóa FP8) vào thư mục `AI-6/weights/` để hệ thống nhúng nạp vào vận hành.
