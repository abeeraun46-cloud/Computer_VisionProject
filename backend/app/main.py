from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from backend.app.routers import image, video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    app.state.detection_model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded")

app.include_router(image.router, prefix="/image")
app.include_router(video.router, prefix="/video")

@app.get("/health")
def health():
    return {"status": "ok"}