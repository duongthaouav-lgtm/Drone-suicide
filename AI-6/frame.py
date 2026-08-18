import cv2
import os

video_path = "VIDEO.MOV"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"frame_{frame_id:05d}.jpg")
    cv2.imwrite(filename, frame)

    frame_id += 1

cap.release()

print("Done! Total frames:", frame_id)