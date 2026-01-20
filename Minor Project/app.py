import streamlit as st
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Traffic Accident Detection",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "sos" not in st.session_state:
    st.session_state.sos = False
    st.session_state.logs = []

# =========================================================
# STRONG CSS (FOR VISIBILITY)
# =========================================================
st.markdown("""
<style>
body { background-color:#020617; }

.sos-box {
    background:#020617;
    border:3px solid red;
    padding:20px;
    border-radius:15px;
    min-height:400px;
}

.sos-active {
    background:red;
    color:white;
    padding:15px;
    text-align:center;
    border-radius:10px;
    font-weight:bold;
    animation:pulse 1.2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,0,0,.7); }
    70% { box-shadow: 0 0 0 20px rgba(255,0,0,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("<h1 style='text-align:center;'>🚨 AI Traffic Accident Detection</h1>", unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([3, 1])

# ================= LEFT =================
with left:
    video = st.file_uploader("📤 Upload Traffic Video", ["mp4", "avi", "mov"])
    start = st.button("▶ Start Detection")

    video_box = st.empty()

# ================= RIGHT (SOS PANEL – ALWAYS VISIBLE) =================
with right:
    st.markdown("## 🚑 SOS PANEL")
    st.markdown('<div class="sos-box">', unsafe_allow_html=True)

    if st.session_state.sos:
        st.markdown('<div class="sos-active">🚨 SOS ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.info("SOS Idle")

    st.markdown("### 📡 Logs")
    if st.session_state.logs:
        for l in st.session_state.logs:
            st.write(l)
    else:
        st.caption("No activity yet")

    if st.button("🚑 MANUAL SOS"):
        st.session_state.sos = True
        st.session_state.logs.append("📤 Manual SOS triggered")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================================================
# DETECTION
# =========================================================
if start and video:
    st.session_state.sos = False
    st.session_state.logs.clear()

    with open("temp.mp4", "wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture("temp.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        if any(r.boxes is not None and len(r.boxes) > 2 for r in results):
            st.session_state.sos = True
            st.session_state.logs.extend([
                "📤 SOS Sent",
                "📥 SOS Received",
                "🚑 Ambulance Dispatched"
            ])
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_box.image(frame, use_container_width=True)
        time.sleep(0.03)

    cap.release()
