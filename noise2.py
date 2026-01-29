# import streamlit as st
# import numpy as np
# from scipy.io import wavfile
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import os
# import joblib
# import tempfile
# import matplotlib.pyplot as plt

# # ================= PAGE CONFIG =================
# st.set_page_config(
#     page_title="Noise Level Analyzer",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.title("🔊 Noise Level Analyzer")
# st.markdown("Analyze audio files and classify noise levels using Machine Learning")

# # ================= FEATURE EXTRACTION =================
# def extract_features(audio_data, sample_rate):
#     audio = audio_data.astype(np.float32)

#     # Normalize PCM
#     if np.issubdtype(audio_data.dtype, np.integer):
#         audio /= np.iinfo(audio_data.dtype).max

#     # Stereo → Mono
#     if audio.ndim > 1:
#         audio = np.mean(audio, axis=1)

#     rms = np.sqrt(np.mean(audio ** 2))
#     zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

#     return [rms, zcr], audio, sample_rate


# def calculate_db(rms):
#     return max(0, int(20 * np.log10(rms + 1e-6) + 60))


# # ================= MODEL =================
# @st.cache_resource
# def load_model():
#     model_path = "saved_model.joblib"

#     if os.path.exists(model_path):
#         data = joblib.load(model_path)
#         return data["model"], data["scaler"], "Loaded trained model"

#     # ---- Synthetic Training ----
#     X, y = [], []
#     sr = 22050
#     t = np.linspace(0, 2, int(sr * 2), endpoint=False)

#     def gen(freq, amp):
#         return amp * np.sin(2 * np.pi * freq * t)

#     classes = [
#         (0, [0.01, 0.02]),   # LOW
#         (1, [0.05, 0.08]),   # MEDIUM
#         (2, [0.15, 0.25])    # HIGH
#     ]

#     for label, amps in classes:
#         for amp in amps:
#             audio = gen(440, amp)
#             features, _, _ = extract_features(audio, sr)
#             X.append(features)
#             y.append(label)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = RandomForestClassifier(
#         n_estimators=50,
#         random_state=42
#     )
#     model.fit(X_scaled, y)

#     joblib.dump({"model": model, "scaler": scaler}, model_path)
#     return model, scaler, "Trained on synthetic audio data"


# def classify_audio(model, scaler, features):
#     features_scaled = scaler.transform([features])
#     pred = model.predict(features_scaled)[0]
#     prob = model.predict_proba(features_scaled)[0][pred] * 100

#     labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
#     return labels[pred], prob


# # ================= SIDEBAR =================
# st.sidebar.header("📋 Model Info")
# model, scaler, status = load_model()
# st.sidebar.success(status)

# st.sidebar.markdown("---")
# st.sidebar.info("""
# **LOW**: Quiet environment  
# **MEDIUM**: Normal speech  
# **HIGH**: Loud noise  
# """)

# # ================= MAIN UI =================
# col1, col2 = st.columns(2)

# with col1:
#     uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(uploaded_file.read())
#             tmp_path = tmp.name

#         sr, audio = wavfile.read(tmp_path)
#         features, audio_mono, _ = extract_features(audio, sr)

#         rms, zcr = features
#         db_value = calculate_db(rms)

#         label, confidence = classify_audio(model, scaler, features)

#         os.unlink(tmp_path)

#         with col2:
#             st.subheader("📊 Results")

#             st.audio(uploaded_file)

#             color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
#             st.markdown(f"## {color[label]} {label}")

#             c1, c2, c3 = st.columns(3)
#             c1.metric("Sound Level", f"{db_value} dB")
#             c2.metric("RMS", f"{rms:.4f}")
#             c3.metric("Confidence", f"{confidence:.2f}%")

#         # ===== Waveform =====
#         st.markdown("---")
#         st.subheader("🌊 Waveform")

#         fig, ax = plt.subplots(figsize=(12, 3))
#         time = np.linspace(0, len(audio_mono) / sr, len(audio_mono))
#         ax.plot(time, audio_mono)
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Amplitude")
#         st.pyplot(fig)

#         # ===== Precautions =====
#         st.markdown("---")
#         st.subheader("⚠️ Safety Recommendations")

#         if label == "LOW":
#             st.success("Safe environment. No precautions needed.")
#         elif label == "MEDIUM":
#             st.warning("Limit long exposure. Take breaks.")
#         else:
#             st.error("High noise! Use hearing protection.")

#     else:
#         st.info("Upload a WAV file to begin analysis")

# # ================= FOOTER =================
# st.markdown("""
# <div style="text-align:center;color:gray;font-size:12px">
# Noise Level Analyzer | ML-based RMS & ZCR Classification
# </div>
# """, unsafe_allow_html=True)





import streamlit as st
import numpy as np
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib
import tempfile
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Noise Level Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔊 Noise Level Analyzer")
st.markdown("Analyze audio files and classify environmental noise levels")

# ================= FEATURE EXTRACTION =================
def extract_features(audio_data, sample_rate):
    audio = audio_data.astype(np.float32)

    if np.issubdtype(audio_data.dtype, np.integer):
        audio /= np.iinfo(audio_data.dtype).max

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    rms = np.sqrt(np.mean(audio ** 2))
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

    return [rms, zcr], audio, sample_rate


def calculate_db(rms):
    return max(0, int(20 * np.log10(rms + 1e-6) + 60))


# ================= MODEL =================
@st.cache_resource
def load_model():
    model_path = "saved_model.joblib"

    if os.path.exists(model_path):
        data = joblib.load(model_path)
        return data["model"], data["scaler"], "Loaded trained model"

    X, y = [], []
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2), endpoint=False)

    def gen(freq, amp):
        return amp * np.sin(2 * np.pi * freq * t)

    classes = [
        (0, [0.01, 0.02]),   # LOW
        (1, [0.05, 0.08]),   # MEDIUM
        (2, [0.15, 0.25])    # HIGH
    ]

    for label, amps in classes:
        for amp in amps:
            audio = gen(440, amp)
            features, _, _ = extract_features(audio, sr)
            X.append(features)
            y.append(label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump({"model": model, "scaler": scaler}, model_path)
    return model, scaler, "Trained on synthetic audio data"


def classify_audio(model, scaler, features):
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return labels[pred]


# ================= SIDEBAR =================
st.sidebar.header("📋 Model Info")
model, scaler, status = load_model()
st.sidebar.success(status)

st.sidebar.markdown("---")
st.sidebar.info("""
**LOW**: Quiet environment  
**MEDIUM**: Normal speech level  
**HIGH**: Loud / harmful noise  
""")

# ================= MAIN UI =================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        sr, audio = wavfile.read(tmp_path)
        features, audio_mono, _ = extract_features(audio, sr)

        rms, zcr = features
        db_value = calculate_db(rms)

        label = classify_audio(model, scaler, features)
        os.unlink(tmp_path)

        with col2:
            st.subheader("📊 Analysis Result")

            st.audio(uploaded_file)

            color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            st.markdown(f"## {color[label]} {label}")

            c1, c2 = st.columns(2)
            c1.metric("Sound Level", f"{db_value} dB")
            c2.metric("RMS Value", f"{rms:.4f}")

        # ===== Waveform =====
        st.markdown("---")
        st.subheader("🌊 Audio Waveform")

        fig, ax = plt.subplots(figsize=(12, 3))
        time = np.linspace(0, len(audio_mono) / sr, len(audio_mono))
        ax.plot(time, audio_mono)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # ===== Precautions =====
        st.markdown("---")
        st.subheader("⚠️ Precautions & Safety Guidelines")

        if label == "LOW":
            st.success("""
            ✅ **Low Noise Level**
            - Safe for long-term exposure  
            - Ideal for studying and office work  
            - No hearing risk  
            - Suitable for libraries, homes, classrooms  
            """)

        elif label == "MEDIUM":
            st.warning("""
            ⚠️ **Moderate Noise Level**
            - Prolonged exposure may cause tiredness  
            - Take short breaks every 1–2 hours  
            - Avoid using high-volume headphones  
            - Suitable for streets, offices, conversations  
            """)

        else:
            st.error("""
            🔴 **High Noise Level – Dangerous**
            - Risk of hearing damage  
            - Use earplugs or noise-cancelling headphones  
            - Limit exposure to less than 1 hour  
            - Avoid continuous listening  
            - Common in traffic, construction sites, loud machinery  
            """)

    else:
        st.info("Upload a WAV file to start analysis")

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center;color:gray;font-size:12px">
Noise Level Analyzer | ML-based Audio Classification
</div>
""", unsafe_allow_html=True)
