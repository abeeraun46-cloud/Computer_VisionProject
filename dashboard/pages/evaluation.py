import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import matplotlib.pyplot as plt

from ui import load_css

# ------------------ إعداد الصفحة ------------------
st.set_page_config(layout="wide")
load_css()

# ------------------ تحميل النموذج ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------ Tracker ------------------
class SimpleTracker:
    def __init__(self, max_distance=70, max_missing=25):
        self.objects = {}
        self.missing = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, detections):
        updated = {}
        results = []

        centers = []
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy, x1, y1, x2, y2))

        used = set()

        for pid, (px, py) in self.objects.items():
            best_match = None
            best_dist = 9999

            for i, (cx, cy, x1, y1, x2, y2) in enumerate(centers):
                if i in used:
                    continue

                dist = math.hypot(cx - px, cy - py)
                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = (i, cx, cy, x1, y1, x2, y2)

            if best_match:
                i, cx, cy, x1, y1, x2, y2 = best_match
                used.add(i)
                updated[pid] = (cx, cy)
                self.missing[pid] = 0
                results.append(pid)

        for i, (cx, cy, x1, y1, x2, y2) in enumerate(centers):
            if i not in used:
                pid = self.next_id
                self.next_id += 1
                updated[pid] = (cx, cy)
                self.missing[pid] = 0
                results.append(pid)

        for pid in list(self.objects.keys()):
            if pid not in updated:
                self.missing[pid] += 1
                if self.missing[pid] < self.max_missing:
                    updated[pid] = self.objects[pid]

        self.objects = updated
        return results

# ------------------ التطبيق ------------------
def run():

    st.markdown("""
    <div class="card">
        <h1>📈 Model Evaluation</h1>
        <p>Evaluate real system performance based on live inference.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload evaluation video", type=["mp4", "avi", "mov"])

    if uploaded is None:
        st.info("Upload a video to run evaluation")
        return

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(uploaded.read())

    st.video(temp.name)

    if not st.button("▶ Run Evaluation"):
        return

    cap = cv2.VideoCapture(temp.name)
    tracker = SimpleTracker()

    frame_times = []
    people_over_time = []
    unique_ids = set()

    frame_count = 0
    start_total = time.time()

    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start = time.time()

        results = model(frame, conf=0.4, classes=[0], verbose=False)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))

        tracked_ids = tracker.update(detections)

        for pid in tracked_ids:
            unique_ids.add(pid)

        current_people = len(tracked_ids)

        end = time.time()
        frame_times.append(end - start)
        people_over_time.append(current_people)

        progress.progress(min(frame_count / 300, 1.0))

    cap.release()

    total_time = time.time() - start_total

    # ------------------ Metrics ------------------
    avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
    avg_frame_time = np.mean(frame_times) if frame_times else 0
    max_people = max(people_over_time) if people_over_time else 0

    # ------------------ عرض النتائج ------------------
    st.markdown(f"""
    <div class="card">
        <h2>📊 Evaluation Metrics</h2>
        <div class="stats-list">
            <div class="stat-row"><span class="stat-label">Total Frames</span><span class="stat-value">{len(frame_times)}</span></div>
            <div class="stat-row"><span class="stat-label">Average FPS</span><span class="stat-value">{avg_fps:.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Avg Frame Time (ms)</span><span class="stat-value">{avg_frame_time*1000:.1f}</span></div>
            <div class="stat-row"><span class="stat-label">Max People Detected</span><span class="stat-value">{max_people}</span></div>
            <div class="stat-row"><span class="stat-label">Unique IDs</span><span class="stat-value">{len(unique_ids)}</span></div>
            <div class="stat-row"><span class="stat-label">Total Processing Time (s)</span><span class="stat-value">{total_time:.1f}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------ رسم عدد الأشخاص ------------------
    st.markdown("<div class='card'><h3>👥 People Count Over Time</h3>", unsafe_allow_html=True)

    fig1 = plt.figure()
    plt.plot(people_over_time)
    plt.xlabel("Frame")
    plt.ylabel("People Count")
    plt.title("Crowd Over Time")
    plt.grid(alpha=0.3)

    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ رسم زمن المعالجة ------------------
    st.markdown("<div class='card'><h3>⚡ Frame Processing Time</h3>", unsafe_allow_html=True)

    fig2 = plt.figure()
    plt.plot(frame_times)
    plt.xlabel("Frame")
    plt.ylabel("Seconds")
    plt.title("Processing Time per Frame")
    plt.grid(alpha=0.3)

    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    st.success("Evaluation completed successfully")

run()
