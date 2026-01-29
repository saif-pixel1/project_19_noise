# 🔊 Noise Level Analyzer - Streamlit App

A web-based audio analysis tool that classifies noise levels and provides dB measurements with visual feedback.

## Features ✨

- **Upload .wav files** for instant analysis
- **dB Level Detection** with precise measurements
- **Three-Level Classification**: Low, Medium, High
- **Confidence Percentages** for each classification
- **Visual Waveform Display** of the audio
- **Safety Precautions** based on noise level
- **Trained on**: high1, high2, low1, low2, med1, med2

## Installation 📦

### Prerequisites
- Python 3.8 or higher

### Setup Steps

1. **Navigate to project folder**:
   ```bash
   cd "c:\Users\Admin\OneDrive\Desktop\noise project"
   ```

2. **Create virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App 🚀

### Local Testing
```bash
streamlit run streamlit_app.py
```

The app will open at: `http://localhost:8501`

### Deploy on Streamlit Cloud (Free)

1. **Create GitHub repository** with your project files
2. **Go to**: https://share.streamlit.io
3. **Connect your GitHub repo**
4. **Select**:
   - Repository: `your-username/repo-name`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. **Click Deploy!**

Your app will be live at: `https://your-app-name.streamlit.app`

## How to Use 📊

1. **Upload a .wav file** using the file uploader
2. **View Results**:
   - Classification (LOW / MEDIUM / HIGH)
   - dB Level reading
   - Confidence percentage
   - Full confidence breakdown chart
3. **See Recommendations** for safe listening practices
4. **View Waveform** to visualize audio pattern

## Classification Guide 📈

| Level | dB Range | Description | Risk Level |
|-------|----------|-------------|-----------|
| 🟢 LOW | 0-20 dB | Very quiet | Safe |
| 🟡 MEDIUM | 20-50 dB | Normal conversation | Moderate |
| 🔴 HIGH | >50 dB | Loud/Dangerous | High |

## Training Data 📁

The model is trained on these audio files in the `audio/` folder:
- `high1.wav` - High noise sample 1
- `high2.wav` - High noise sample 2
- `low1.wav` - Low noise sample 1
- `low2.wav` - Low noise sample 2
- `med1.wav` - Medium noise sample 1
- `med2.wav` - Medium noise sample 2

Test files: `test1.wav` - `test9.wav`

## Metrics Explained 📐

- **dB (Decibels)**: Measurement of sound intensity
- **RMS (Root Mean Square)**: Loudness/amplitude of audio
- **ZCR (Zero Crossing Rate)**: Frequency characteristics
- **Confidence**: How certain the model is about the classification

## Troubleshooting 🔧

### Model not loading?
- Ensure `audio/` folder has training files (high1, high2, low1, low2, med1, med2)
- All files must be `.wav` format

### Upload button not working?
- Check file is `.wav` format
- File size should be reasonable (< 50MB)

### App running slow?
- First load caches the model (subsequent loads are faster)
- Streamlit Cloud may take 30 seconds for first load

## Files Structure 📁

```
noise project/
├── streamlit_app.py      (Main Streamlit app)
├── noise_classifier.py   (Training & classification logic)
├── requirements.txt      (Python dependencies)
├── README.md            (This file)
└── audio/
    ├── high1.wav
    ├── high2.wav
    ├── low1.wav
    ├── low2.wav
    ├── med1.wav
    ├── med2.wav
    └── test1-9.wav
```

## Author Notes 📝

- Model uses Random Forest classifier for noise classification
- Features extracted: RMS and Zero Crossing Rate
- Trained with provided audio samples
- Automatically handles mono and stereo audio

---

**Enjoy analyzing your audio!** 🎵
