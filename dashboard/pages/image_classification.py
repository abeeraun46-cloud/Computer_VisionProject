import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import streamlit.components.v1 as components

from ui import load_css

# ---------------- إعداد الصفحة ----------------
st.set_page_config(layout="wide")
load_css()

st.markdown("""
<div class="card">
    <h1>🧠 Image Classification</h1>
    <p>Upload an image and run YOLOv8 classification with multi-class prediction.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- تحميل النموذج ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n-cls.pt")

model = load_model()

# ---------------- الواجهة ----------------
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Run Classification"):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)

        # تشغيل النموذج
        results = model(image)

        probs = results[0].probs
        names = results[0].names

        top5 = probs.top5
        top5conf = probs.top5conf

        # بناء HTML ديناميكي
        bars_html = ""

        for i, cls_id in enumerate(top5):
            label = names[cls_id]
            conf = float(top5conf[i]) * 100

            bars_html += f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value">{conf:.2f}%</span>
            </div>

            <div style="background:#222;border-radius:8px;overflow:hidden;margin-bottom:12px">
                <div style="
                    width:{conf}%;
                    height:10px;
                    background:linear-gradient(90deg,#00c6ff,#0072ff);
                    transition:0.5s;
                "></div>
            </div>
            """

        # عرض النتيجة
        components.html(f"""
        <div class="card">
            <h2>📌 Prediction Result (Top-5)</h2>
            <div class="stats-list">
                {bars_html}
            </div>
        </div>
        """, height=500)

        st.success("Classification completed successfully")