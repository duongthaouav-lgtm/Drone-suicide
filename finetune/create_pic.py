import os

# đường dẫn tới folder ảnh
image_dir = "cv dataset"

# folder label sẽ tạo
label_dir = os.path.join(image_dir, "labels")

# tạo folder labels nếu chưa có
os.makedirs(label_dir, exist_ok=True)

# các định dạng ảnh phổ biến
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)

    # kiểm tra có phải file ảnh không
    if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_extensions:
        
        # đổi tên thành .txt
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)

        # tạo file rỗng
        open(txt_path, "w").close()

print("Done! Created empty label files.")