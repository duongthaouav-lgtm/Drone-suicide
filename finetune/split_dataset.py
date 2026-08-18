import os
import random
import shutil

# nguồn background
src_images = "test/images"
src_labels = "test/labels"

# dataset chính
dst_images = "dataset/images"
dst_labels = "dataset/labels"

# số lượng background muốn thêm
num_train = 200
num_val = 60
num_test = 40

# lấy danh sách ảnh
images = [f for f in os.listdir(src_images) if f.endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

train_imgs = images[:num_train]
val_imgs = images[num_train:num_train+num_val]
test_imgs = images[num_train+num_val:num_train+num_val+num_test]


def copy_files(file_list, split):

    for img in file_list:

        label = img.rsplit(".", 1)[0] + ".txt"

        src_img_path = os.path.join(src_images, img)
        src_lbl_path = os.path.join(src_labels, label)

        dst_img_path = os.path.join(dst_images, split, img)
        dst_lbl_path = os.path.join(dst_labels, split, label)

        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_lbl_path, dst_lbl_path)


copy_files(train_imgs, "train")
copy_files(val_imgs, "val")
copy_files(test_imgs, "test")

print("Background images added successfully")