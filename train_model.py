from ultralytics import YOLO

# تحميل نموذج جاهز كأساس
model = YOLO("yolov8n-cls.pt")

# تدريب على بياناتك
model.train(
    data="dataset",
    epochs=15,
    imgsz=224,
    batch=8
)