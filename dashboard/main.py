import streamlit as st
from ui import load_css

st.set_page_config(page_title="Smart Vision Dashboard", layout="wide")
load_css()

st.markdown("""
<div class="header">
    <h1>👁 Smart Vision Dashboard</h1>
    <p>AI-powered system for video understanding, behavior analysis, and crowd intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>📦 Available Modules</h2>
    <ul>
        <li>🎥 Video Analysis (Tracking, Heatmap, Crowd Alert)</li>
        <li>🖼 Image Classification</li>
        <li>🧩 Segmentation</li>
        <li>📊 Model Evaluation</li>
        <li>📱 Phone Usage Detection (New)</li>
    </ul>
    <p>➡ Use sidebar to navigate between modules</p>
</div>
""", unsafe_allow_html=True)