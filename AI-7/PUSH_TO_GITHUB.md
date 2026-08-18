# 🚀 Hướng Dẫn Push Code Lên GitHub

## 📋 Chuẩn Bị

### Bước 1: Kiểm tra Git đã được cài đặt

```bash
git --version
```

Nếu chưa có, tải từ: https://git-scm.com/download/win

### Bước 2: Cấu hình Git (nếu chưa có)

```bash
git config --global user.name "dkzdragon02"
git config --global user.email "your-email@example.com"
```

## 🔧 Các Bước Push Code

### Bước 1: Khởi tạo Git Repository

```bash
cd C:\Users\CuaDa1De\Desktop\AI\AI-7
git init
```

### Bước 2: Thêm Remote Repository

```bash
git remote add origin https://github.com/dkzdragon02/AI-UAV-DRONE.git
```

### Bước 3: Kiểm tra các file sẽ được commit

```bash
git status
```

### Bước 4: Thêm tất cả các file (trừ những file trong .gitignore)

```bash
git add .
```

### Bước 5: Commit

```bash
git commit -m "Initial commit: Drone detection system with YOLO"
```

### Bước 6: Đổi tên branch chính (nếu cần)

```bash
git branch -M main
```

### Bước 7: Push lên GitHub

```bash
git push -u origin main
```

**Lưu ý:** Lần đầu push sẽ yêu cầu đăng nhập GitHub:
- Username: `dkzdragon02`
- Password: Sử dụng **Personal Access Token** (không phải password thường)

### Tạo Personal Access Token

1. Vào GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Chọn quyền: `repo` (full control)
4. Copy token và dùng làm password khi push

## 📝 Script Tự Động (PowerShell)

Tạo file `push_to_github.ps1`:

```powershell
# Khởi tạo Git
git init

# Thêm remote
git remote add origin https://github.com/dkzdragon02/AI-UAV-DRONE.git

# Thêm tất cả files
git add .

# Commit
git commit -m "Initial commit: Drone detection system with YOLO"

# Đổi tên branch
git branch -M main

# Push
git push -u origin main
```

Chạy script:
```powershell
.\push_to_github.ps1
```

## ⚠️ Lưu Ý

### Files KHÔNG được push (đã có trong .gitignore):

- ✅ `weights/*.pt` - Model files (quá lớn)
- ✅ `results/` - Logs và videos
- ✅ `datasets/` - Dataset files (quá lớn)
- ✅ `*.mp4`, `*.MOV` - Video files
- ✅ `__pycache__/` - Python cache
- ✅ `.venv/` - Virtual environment

### Files ĐƯỢC push:

- ✅ `test_models.py`
- ✅ `download_model.py`
- ✅ `download_and_prepare_dataset.py`
- ✅ `train_drone_model.py`
- ✅ `README.md`
- ✅ `.gitignore`
- ✅ `HUONG_DAN_DATASET.md`

## 🔄 Cập Nhật Code Sau Này

Khi có thay đổi:

```bash
git add .
git commit -m "Mô tả thay đổi"
git push
```

## 🐛 Troubleshooting

### Lỗi: "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/dkzdragon02/AI-UAV-DRONE.git
```

### Lỗi: "Authentication failed"

- Kiểm tra Personal Access Token
- Hoặc sử dụng SSH: `git remote set-url origin git@github.com:dkzdragon02/AI-UAV-DRONE.git`

### Lỗi: "Large files"

Nếu có file quá lớn (>100MB), GitHub sẽ từ chối. Kiểm tra `.gitignore` đã loại bỏ:
- Model weights (*.pt)
- Videos (*.mp4, *.MOV)
- Dataset files

## ✅ Kiểm Tra

Sau khi push thành công, truy cập:
https://github.com/dkzdragon02/AI-UAV-DRONE

Bạn sẽ thấy tất cả code đã được upload!

