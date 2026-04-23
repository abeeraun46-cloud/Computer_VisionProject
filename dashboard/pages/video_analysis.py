import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

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
                results.append((pid, x1, y1, x2, y2))

        for i, (cx, cy, x1, y1, x2, y2) in enumerate(centers):
            if i not in used:
                pid = self.next_id
                self.next_id += 1
                updated[pid] = (cx, cy)
                self.missing[pid] = 0
                results.append((pid, x1, y1, x2, y2))

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
        <h1>🎥 Smart Crowd & Behavior Analysis</h1>
        <p>Upload a surveillance video and run real-time people analysis, tracking, heatmap, and crowd alerts.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload surveillance video", type=["mp4", "avi", "mov"])

    if uploaded is None:
        st.info("Upload a video to start analysis")
        return

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(uploaded.read())

    st.video(temp.name)

    col_btn1, col_btn2 = st.columns(2)

    start = col_btn1.button("▶ Start Analysis")
    stop = col_btn2.button("⛔ Stop Analysis")

    if "stop_flag" not in st.session_state:
        st.session_state.stop_flag = False

    if stop:
        st.session_state.stop_flag = True

    if start:
        st.session_state.stop_flag = False

        cap = cv2.VideoCapture(temp.name)

        col1, col2 = st.columns([2, 1])
        frame_box = col1.empty()
        stats_box = col2.empty()

        tracker = SimpleTracker()
        history = defaultdict(list)
        heatmap = None

        id_confirm_frames = defaultdict(int)
        confirmed_ids = set()
        CONFIRM_THRESHOLD = 15

        LINE_Y = None
        counted_ids = set()
        entered_count = 0

        frame_count = 0
        CROWD_THRESHOLD = 6

        while True:
            if st.session_state.stop_flag:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            if LINE_Y is None:
                LINE_Y = h // 2

            if heatmap is None:
                heatmap = np.zeros((h, w), dtype=np.float32)

            results = model(frame, conf=0.4, classes=[0], verbose=False)

            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2))

            tracked = tracker.update(detections)
            current_people = len(tracked)

            for pid, x1, y1, x2, y2 in tracked:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                id_confirm_frames[pid] += 1
                if id_confirm_frames[pid] == CONFIRM_THRESHOLD:
                    confirmed_ids.add(pid)

                history[pid].append((cx, cy))
                heatmap[cy, cx] += 1

                status = "Normal"
                if len(history[pid]) > 25:
                    dx = abs(history[pid][-1][0] - history[pid][-25][0])
                    dy = abs(history[pid][-1][1] - history[pid][-25][1])
                    if dx + dy < 12:
                        status = "Long Stay"
                    else:
                        status = "Moving"

                if pid in confirmed_ids and pid not in counted_ids:
                    if cy > LINE_Y:
                        counted_ids.add(pid)
                        entered_count += 1

                color = (0,255,0) if status == "Normal" else \
                        (0,165,255) if status == "Moving" else \
                        (0,0,255)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, status, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255,255,255), 2)

            crowd_alert = current_people >= CROWD_THRESHOLD

            stats_box.markdown(f"""
            <div class="card">
            <h3>📊 Live Analytics</h3>

            <div class="stat-row">
                <span class="stat-label">Frame</span>
                <span class="stat-value">{frame_count}</span>
            </div>

            <div class="stat-row">
                <span class="stat-label">Current People</span>
                <span class="stat-value">{current_people}</span>
            </div>

            <div class="stat-row">
                <span class="stat-label">Line Crossed</span>
                <span class="stat-value">{entered_count}</span>
            </div>

            <p>Crowd Alert: {"🚨 YES" if crowd_alert else "✅ No"}</p>
            </div>
            """, unsafe_allow_html=True)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame, use_container_width=True)

            time.sleep(0.01)

        cap.release()

        st.markdown("<div class='card'><h2>🔥 Crowd Density Heatmap</h2>", unsafe_allow_html=True)

        heatmap_blur = cv2.GaussianBlur(heatmap, (61,61), 0)
        heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

        st.image(heatmap_color, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.success("Analysis completed successfully")


run()