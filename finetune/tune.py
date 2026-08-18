from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    model.tune(
        data="dataset.yaml",
        epochs=35,
        imgsz=640,
        batch=32,
        device=0,
        workers=4,
        iterations=15,

        # training params (khóa lại)
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=3,
        warmup_momentum=0.8,

        # loss params (khóa lại)
        box=7.5,    
        cls=0.5,
        dfl=1.5,

        # augmentation KHÓA
        degrees=0.0,
        translate=0.1,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        copy_paste=0.0,
        close_mosaic=10,
        # KHÔNG truyền:
        # mosaic
        # mixup
        # hsv_h
        # hsv_s
        # hsv_v

        project="runs",
        name="drone_detector_tune"
    )

if __name__ == "__main__":
    main()