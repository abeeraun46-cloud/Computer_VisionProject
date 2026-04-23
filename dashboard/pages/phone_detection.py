import streamlit as st
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components
from ui import load_css

st.set_page_config(layout="wide")
load_css()

st.markdown("""
<div class="card">
    <h1>📱 Phone Usage Detection</h1>
    <p>Upload an image and detect whether the person is holding a phone.</p>
</div>
""", unsafe_allow_html=True)

# تحميل النموذج المدرب
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\lenovo\Desktop\مشروع نهائيCOM VS\runs\classify\train2\weights\best.pt")

model = load_model()

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Run Detection"):
        results = model(image)

        probs = results[0].probs
        names = results[0].names

        html = "<div class='card'><h2>📊 Prediction</h2>"

        for i, conf in enumerate(probs.data.tolist()):
            label = names[i]
            percent = conf * 100

            html += f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value">{percent:.2f}%</span>
            </div>

            <div style="background:#222;border-radius:8px;overflow:hidden;margin-bottom:12px">
                <div style="
                    width:{percent}%;
                    height:12px;
                    background:linear-gradient(90deg,#00c6ff,#0072ff);
                "></div>
            </div>
            """

        html += "</div>"

        components.html(html, height=400)
        st.success("Done")