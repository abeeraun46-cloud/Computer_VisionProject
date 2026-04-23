from fastapi import UploadFile
from PIL import Image
import io
import numpy as np

# ------------------- تحويل UploadFile إلى صورة -------------------
async def read_image_file(file: UploadFile) -> Image.Image:
    """Reads an uploaded file and converts it to a PIL Image."""
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

# ------------------- تجهيز نتائج الكشف -------------------
def format_detections(results, model_names):
    """Formats YOLO detection results into a JSON-friendly structure."""
    detections = []
    for r in results:
        for box in r.boxes:
            label = model_names[int(box.cls)]
            conf = float(box.conf)
            bbox = box.xyxy.tolist()[0]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": bbox
            })
    return detections

# ------------------- تجهيز نتائج التجزئة -------------------
def format_segmentation(results):
    """Extracts masks from YOLO segmentation results."""
    masks = []
    for r in results:
        if hasattr(r, "masks") and r.masks is not None:
            masks.extend(r.masks.data.tolist())
    return masks

# ------------------- حساب متوسط عدد الأشخاص في الفيديو -------------------
def average_people_count(analytics):
    """Calculates average people count from video analytics."""
    if not analytics:
        return 0
    return float(np.mean([a["people_count"] for a in analytics]))