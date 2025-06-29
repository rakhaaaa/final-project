import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import datetime
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ===== PAGE CONFIGURATION =====
st.set_page_config(page_title="Face Emotion & Identity Analyzer", layout="wide", page_icon="🎭")
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #dbeafe, #f3e8ff);
            color: #1e1e2f;
        }
        .stButton>button {
            color: white;
            background: #6366F1;
            padding: 0.6em 1.2em;
            border: none;
            border-radius: 0.5em;
            font-weight: 600;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: #4F46E5;
        }
        .stRadio>div>label {
            font-weight: 600;
        }
        .block-container {
            padding-top: 2rem;
        }
        .emoji {
            font-size: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("😄 Face Emotion & Identity Analyzer")
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <span class='emoji'>📸</span>
        <h4 style='margin-left: 1rem;'>Analisis ekspresi wajah dari gambar dan webcam secara real-time menggunakan DeepFace</h4>
    </div>
    <hr style='margin-top: 1rem; margin-bottom: 2rem;'>
""", unsafe_allow_html=True)

# ===== LOGGING CONFIGURATION =====
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "face_analysis_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_COLUMNS = ["Waktu", "Emosi", "Kepercayaan Emosi", "Usia", "Gender", "Ras"]

def simpan_log(data):
    df = pd.DataFrame(data, columns=LOG_COLUMNS)
    df.fillna("", inplace=True)  # jaga agar kolom tidak hilang
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)


def tampilkan_log():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, names=LOG_COLUMNS, header=0, on_bad_lines='skip')
            st.subheader("📄 Riwayat Analisis")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membaca log: {e}")
    else:
        st.info("Belum ada histori analisis.")

# ===== WEBCAM EMOTION DETECTOR =====
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            emotion = max(result["emotion"], key=result["emotion"].get)
            cv2.putText(img, f"Emosi: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        except Exception as e:
            print("Error:", e)
        return img

# ===== MODE SELECTOR =====
mode_input = st.radio("📷 Pilih Metode Input", ["Upload Gambar", "Deteksi Webcam Live"], horizontal=True)
show_details = st.toggle("🧠 Tampilkan info tambahan (usia, gender, ras)")

# ===== MODE: UPLOAD IMAGE =====
if mode_input == "Upload Gambar":
    uploaded_file = st.file_uploader("📤 Upload Foto Wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Foto yang Diupload", use_container_width=True)

        img_array = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        actions = ["emotion"] + (["age", "gender", "race"] if show_details else [])

        st.info(f"🔍 Menganalisis: {', '.join(actions)}")

        with st.spinner("⏳ Menganalisis wajah..."):
            try:
                result = DeepFace.analyze(img_bgr, actions=actions, enforce_detection=False)
                if not isinstance(result, list):
                    result = [result]

                log_entries = []

                for idx, face in enumerate(result):
                    with st.container():
                        st.markdown(f"### 👤 Wajah #{idx+1}")
                        col1, col2, col3 = st.columns(3)
                        log_data = {col: None for col in LOG_COLUMNS}
                        log_data["Waktu"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        for key in actions:
                            value = face.get(key)
                            if isinstance(value, dict):
                                emotion = max(value, key=value.get)
                                col1.metric(label="😊 Emosi", value=emotion, delta=f"{value[emotion]:.1f}%")
                                log_data["Emosi"] = emotion
                                log_data["Kepercayaan Emosi"] = round(value[emotion], 2)
                            else:
                                if key == "age":
                                    col2.metric("🎂 Usia", value)
                                    log_data["Usia"] = value
                                elif key == "gender":
                                    col3.metric("🚻 Gender", value)
                                    log_data["Gender"] = value
                                elif key == "race":
                                    r = max(value, key=value.get)
                                    col1.metric("🌍 Ras", r)
                                    log_data["Ras"] = r

                        log_entries.append(log_data)

                simpan_log(log_entries)

            except Exception as e:
                st.error(f"❌ Gagal analisis: {e}")

# ===== MODE: WEBCAM LIVE =====
elif mode_input == "Deteksi Webcam Live":
    st.info("📹 Akses webcam akan dimulai di bawah. Tekan 'Stop' untuk berhenti.")
    webrtc_streamer(
        key="emotion-stream",
        video_processor_factory=EmotionDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ===== SHOW LOG HISTORY =====
with st.expander("📂 Lihat Riwayat Analisis"):
    tampilkan_log()
