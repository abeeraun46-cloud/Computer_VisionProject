from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from PIL import Image
import io

router = APIRouter()

# ------------------- Object Detection -------------------
@router.post("/detect")
async def detect_image(request: Request, file: UploadFile = File(...)):
    model = getattr(request.app.state, "detection_model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Detection model not loaded")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model.predict(source=img,conf=0.4, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy.tolist()[0]
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox
                })

        return {"detections": detections, "total": len(detections)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in detection: {e}")


# ------------------- Image Segmentation -------------------
@router.post("/segment")
async def segment_image(request: Request, file: UploadFile = File(...)):
    model = getattr(request.app.state, "segmentation_model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Segmentation model not loaded")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model.predict(source=img, conf=0.4, verbose=False)
        masks = []
        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                masks.extend(r.masks.data.tolist())

        return {
            "num_masks": len(masks),
            "has_masks": bool(masks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in segmentation: {e}")