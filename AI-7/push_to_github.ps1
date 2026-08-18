# Script tự động push code lên GitHub
# Chạy: .\push_to_github.ps1

Write-Host "=" -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "🚀 PUSH CODE LÊN GITHUB" -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Git
try {
    $gitVersion = git --version
    Write-Host "✅ Git đã được cài đặt: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git chưa được cài đặt!" -ForegroundColor Red
    Write-Host "   Tải từ: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Kiểm tra đã có .git chưa
if (Test-Path ".git") {
    Write-Host "✅ Git repository đã được khởi tạo" -ForegroundColor Green
} else {
    Write-Host "📦 Đang khởi tạo Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Đã khởi tạo Git repository" -ForegroundColor Green
}

# Kiểm tra remote
$remoteExists = git remote get-url origin 2>$null
if ($remoteExists) {
    Write-Host "✅ Remote 'origin' đã tồn tại: $remoteExists" -ForegroundColor Green
    $update = Read-Host "   Có muốn cập nhật remote? (y/n)"
    if ($update -eq "y") {
        git remote remove origin
        git remote add origin https://github.com/dkzdragon02/AI-UAV-DRONE.git
        Write-Host "✅ Đã cập nhật remote" -ForegroundColor Green
    }
} else {
    Write-Host "📡 Đang thêm remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/dkzdragon02/AI-UAV-DRONE.git
    Write-Host "✅ Đã thêm remote" -ForegroundColor Green
}

# Kiểm tra .gitignore
if (Test-Path ".gitignore") {
    Write-Host "✅ File .gitignore đã tồn tại" -ForegroundColor Green
} else {
    Write-Host "⚠️  File .gitignore chưa tồn tại!" -ForegroundColor Yellow
    Write-Host "   Tạo file .gitignore..." -ForegroundColor Yellow
}

# Kiểm tra README.md
if (Test-Path "README.md") {
    Write-Host "✅ File README.md đã tồn tại" -ForegroundColor Green
} else {
    Write-Host "⚠️  File README.md chưa tồn tại!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📋 Đang kiểm tra các file sẽ được commit..." -ForegroundColor Yellow
git status

Write-Host ""
$confirm = Read-Host "Tiếp tục commit và push? (y/n)"
if ($confirm -ne "y") {
    Write-Host "❌ Đã hủy" -ForegroundColor Red
    exit 0
}

# Thêm tất cả files
Write-Host ""
Write-Host "📦 Đang thêm files..." -ForegroundColor Yellow
git add .

# Commit
Write-Host "💾 Đang commit..." -ForegroundColor Yellow
$commitMessage = Read-Host "Nhập commit message (Enter để dùng mặc định)"
if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Initial commit: Drone detection system with YOLO"
}
git commit -m $commitMessage

# Đổi tên branch
Write-Host "🌿 Đang đổi tên branch thành 'main'..." -ForegroundColor Yellow
git branch -M main

# Push
Write-Host ""
Write-Host "🚀 Đang push lên GitHub..." -ForegroundColor Yellow
Write-Host "   (Có thể yêu cầu đăng nhập GitHub)" -ForegroundColor Cyan
Write-Host ""

try {
    git push -u origin main
    Write-Host ""
    Write-Host "=" -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host "✅ PUSH THÀNH CÔNG!" -ForegroundColor Green
    Write-Host "=" -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔗 Xem repository tại:" -ForegroundColor Cyan
    Write-Host "   https://github.com/dkzdragon02/AI-UAV-DRONE" -ForegroundColor Yellow
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "❌ Lỗi khi push!" -ForegroundColor Red
    Write-Host "   Kiểm tra:" -ForegroundColor Yellow
    Write-Host "   1. Đã đăng nhập GitHub?" -ForegroundColor Yellow
    Write-Host "   2. Personal Access Token đúng chưa?" -ForegroundColor Yellow
    Write-Host "   3. Repository đã tồn tại trên GitHub?" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Xem hướng dẫn trong PUSH_TO_GITHUB.md" -ForegroundColor Cyan
}

