from ultralytics import YOLO

model = YOLO("runs/detect/runs/drone_detector_train5/weights/best.pt")

model.predict(
    source="test/images",
    conf=0.3,
    iou=0.2,
    imgsz=512,
    # visualize=True, 
    save=True
)

# model.predict(
#     source="vid/",
#     conf=0.25,
#     imgsz=960,
#     show=True,
#     save=True
# )