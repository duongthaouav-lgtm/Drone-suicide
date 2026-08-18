from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    model.train(
        data="data.yaml",
        epochs=110,
        imgsz=512,
        batch=64,
        workers=2,
        # lr0             = 0.00166,    # best trial value
        lr0             = 0.001,
        lrf             = 0.01318,
        momentum        = 0.940,      # sweet spot from data
        weight_decay    = 0.0005,     # compromise between 0.0 and 0.001

        # ── Warmup ────────────────────────────────
        warmup_epochs   = 2.78,
        warmup_momentum = 0.95,       # consistently good

        # ── Loss — best trial values ──────────────
        box             = 8.46,       # best trial, avoid >10
        cls             = 0.5,
        dfl             = 1.06,       # lower dfl consistently better

        # ── Augmentation ──────────────────────────
        hsv_h           = 0.0015,        # tuner always pushed to 0
        hsv_s           = 0.807,
        hsv_v           = 0.361,
        degrees         = 0.016,
        translate       = 0.107,
        scale           = 0.5,        # override tuner — too low at 0.233
        shear           = 0.0,
        perspective     = 0.0,
        flipud          = 0.0,
        fliplr          = 0.5,        # override tuner — 0.248 too low
        mosaic          = 0.9,        # tuner confirmed keep at 1.0
        mixup           = 0.0,
        copy_paste      = 0.0,
        close_mosaic    = 10,
        # multi_scale=True,
        # cfg="runs\\detect\\runs\\drone_detector_tune4\\best_hyperparameters.yaml",
        project="runs",
        name="drone_detector_train"
    )   

if __name__ == "__main__":
    main()

