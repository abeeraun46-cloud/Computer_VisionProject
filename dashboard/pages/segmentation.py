import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

from ui import load_css

# ------------------ إعداد الصفحة ------------------
st.set_page_config(layout="wide")
load_css()

# ------------------ تحميل نموذج Segmentation ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n-seg.pt")

model = load_model()

# ------------------ التطبيق ------------------
def run():

    st.markdown("""
    <div class="card">
        <h1>🧩 Image Segmentation</h1>
        <p>Upload an image and apply AI-powered object segmentation using YOLOv8.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Please upload an image to start segmentation.")
        return

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, caption="Original Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Run Segmentation"):
        with st.spinner("Running segmentation..."):

            # تشغيل النموذج
            results = model(img_np, conf=0.4)

            # رسم النتائج
            segmented = results[0].plot()

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(segmented, caption="Segmented Result", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.success("Segmentation completed successfully")

run()