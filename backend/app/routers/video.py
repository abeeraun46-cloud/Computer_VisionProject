from fastapi import APIRouter, UploadFile, File, Request
import tempfile
import cv2
import os
from ultralytics import YOLO

router = APIRouter()

@router.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    try:
        model = request.app.state.detection_model

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "Failed to open video"}

        tracked_ids = set()
        total_unique_people = 0

        results_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # تتبع رسمي بدون lap
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="botsort.yaml",
                conf=0.4
            )

            people = []

            if results and results[0].boxes.id is not None:
                for box, pid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                    pid = int(pid)
                    x1, y1, x2, y2 = map(int, box)

                    if pid not in tracked_ids:
                        tracked_ids.add(pid)
                        total_unique_people += 1

                    people.append({
                        "id": pid,
                        "bbox": [x1, y1, x2, y2]
                    })

            results_frames.append({
                "count": total_unique_people,
                "detections": people
            })

        cap.release()
        os.remove(video_path)

        return {"frames": results_frames}

    except Exception as e:
        return {"error": str(e)}