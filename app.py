import os
# Fix for "TypeError: Descriptors cannot be created directly" on Streamlit Cloud
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
import base64
import io
import wave
import struct
import streamlit.components.v1 as components

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IntelliHeart Pro | Medical AI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PYTHON AUDIO GENERATOR ---
def generate_beep_base64(frequency=880, duration_ms=100):
    """
    Generates a beep sound using pure Python (wave/struct) 
    and returns a Base64-encoded Data URL.
    """
    sample_rate = 44100
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        
        for i in range(n_samples):
            # Sine wave generation with a small decay to avoid clicking
            decay = 1.0 - (i / n_samples)
            value = int(32767.0 * 0.5 * decay * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))
            
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/wav;base64,{b64}"

# Pre-generate sounds in Python
NORMAL_BEEP_B64 = generate_beep_base64(frequency=880, duration_ms=100)
ABNORMAL_BEEP_B64 = generate_beep_base64(frequency=440, duration_ms=300)

# --- PREMIUM CLINICAL UI ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@400;500;600;700&display=swap');
    
    /* === ROOT VARIABLES === */
    :root {
        --bg-primary: #030508;
        --bg-secondary: #0a0f18;
        --bg-card: rgba(12, 18, 28, 0.85);
        --accent-cyan: #00ffd0;
        --accent-red: #ff3e3e;
        --accent-blue: #3b82f6;
        --accent-purple: #a855f7;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --border-glow: rgba(0, 255, 208, 0.15);
        --glass-border: rgba(255, 255, 255, 0.08);
    }
    
    /* === GLOBAL STYLES === */
    .main { 
        background: var(--bg-primary); 
        color: var(--text-primary); 
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp { 
        background: 
            radial-gradient(ellipse at 0% 0%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 100% 0%, rgba(168, 85, 247, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 100%, rgba(0, 255, 208, 0.05) 0%, transparent 50%),
            linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        min-height: 100vh;
    }
    
    /* === HIDE STREAMLIT ELEMENTS === */
    header, footer, #MainMenu { visibility: hidden; }
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }
    
    /* === GLASSMORPHIC CARDS === */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(0, 255, 208, 0.2);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            0 0 20px rgba(0, 255, 208, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    
    /* === METRIC CARDS === */
    .metric-card {
        background: linear-gradient(135deg, rgba(12, 18, 28, 0.9) 0%, rgba(20, 28, 42, 0.85) 100%);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
        padding: 18px 22px;
        border-radius: 14px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan) 0%, var(--accent-blue) 50%, var(--accent-purple) 100%);
        opacity: 0.8;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 255, 208, 0.25);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 1.5px;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'Orbitron', 'JetBrains Mono', monospace;
        font-size: 36px;
        font-weight: 600;
        letter-spacing: 1px;
        line-height: 1.1;
    }
    
    .metric-value-normal { color: var(--accent-cyan); text-shadow: 0 0 20px rgba(0, 255, 208, 0.4); }
    .metric-value-abnormal { color: var(--accent-red); text-shadow: 0 0 20px rgba(255, 62, 62, 0.4); }
    
    /* === STATUS INDICATORS === */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 12px 20px;
        border-radius: 50px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .status-normal {
        background: linear-gradient(135deg, rgba(0, 255, 208, 0.15) 0%, rgba(0, 255, 208, 0.05) 100%);
        border: 1px solid rgba(0, 255, 208, 0.3);
        color: var(--accent-cyan);
        box-shadow: 0 0 20px rgba(0, 255, 208, 0.15);
    }
    
    .status-abnormal {
        background: linear-gradient(135deg, rgba(255, 62, 62, 0.2) 0%, rgba(255, 62, 62, 0.08) 100%);
        border: 1px solid rgba(255, 62, 62, 0.4);
        color: var(--accent-red);
        box-shadow: 0 0 25px rgba(255, 62, 62, 0.2);
        animation: pulse-alert 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse-alert {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 62, 62, 0.2); }
        50% { box-shadow: 0 0 35px rgba(255, 62, 62, 0.4); }
    }
    
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    
    .status-dot-normal { background: var(--accent-cyan); }
    .status-dot-abnormal { background: var(--accent-red); animation-duration: 0.8s; }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    
    /* === HEADER === */
    .main-header {
        text-align: center;
        padding: 25px 0;
        margin-bottom: 20px;
        position: relative;
    }
    
    .header-title {
        font-family: 'Orbitron', 'Inter', sans-serif;
        font-size: 42px;
        font-weight: 600;
        letter-spacing: 8px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #94a3b8 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
    }
    
    .header-title span {
        background: linear-gradient(135deg, var(--accent-red) 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 4px;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-top: 8px;
    }
    
    .header-heartbeat {
        display: inline-block;
        font-size: 32px;
        margin-right: 8px;
        animation: heartbeat 1.2s ease-in-out infinite;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        15% { transform: scale(1.15); }
        30% { transform: scale(1); }
        45% { transform: scale(1.1); }
        60% { transform: scale(1); }
    }
    
    /* === CONFIG PANEL === */
    .config-panel {
        background: linear-gradient(180deg, rgba(12, 18, 28, 0.95) 0%, rgba(8, 12, 20, 0.9) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 24px;
        height: fit-content;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    
    .config-title {
        font-family: 'Orbitron', 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 3px;
        color: var(--text-secondary);
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--glass-border);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .config-title::before {
        content: '⚙️';
        font-size: 16px;
    }
    
    /* === ECG DISPLAY === */
    .ecg-container {
        background: linear-gradient(180deg, rgba(8, 12, 20, 0.98) 0%, rgba(5, 8, 14, 0.95) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 12px 50px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    
    .ecg-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            linear-gradient(90deg, transparent 49.5%, rgba(0, 255, 208, 0.03) 50%, transparent 50.5%),
            linear-gradient(0deg, transparent 49.5%, rgba(0, 255, 208, 0.03) 50%, transparent 50.5%);
        background-size: 40px 40px;
        pointer-events: none;
        opacity: 0.5;
    }
    
    .ecg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        position: relative;
        z-index: 1;
    }
    
    .ecg-title {
        font-family: 'Orbitron', 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 2px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .ecg-live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 62, 62, 0.15);
        border: 1px solid rgba(255, 62, 62, 0.3);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
        color: var(--accent-red);
    }
    
    .ecg-live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-red);
        border-radius: 50%;
        animation: pulse-dot 1s ease-in-out infinite;
    }
    
    /* === STANDBY SCREEN === */
    .standby-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 400px;
        background: linear-gradient(180deg, rgba(8, 12, 20, 0.9) 0%, rgba(5, 8, 14, 0.85) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 60px;
        text-align: center;
    }
    
    .standby-icon {
        font-size: 80px;
        margin-bottom: 24px;
        opacity: 0.4;
        animation: float 4s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    
    .standby-title {
        font-family: 'Orbitron', 'Inter', sans-serif;
        font-size: 24px;
        font-weight: 500;
        letter-spacing: 4px;
        color: var(--text-secondary);
        margin-bottom: 12px;
    }
    
    .standby-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 2px;
        color: rgba(148, 163, 184, 0.6);
        text-transform: uppercase;
    }
    
    /* === STREAMLIT WIDGET OVERRIDES === */
    .stSelectbox > div > div {
        background: rgba(12, 18, 28, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(0, 255, 208, 0.3) !important;
    }
    
    div[data-testid="stMetric"] {
        background: transparent !important;
    }
    
    /* Toggle Switch Enhancement */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
        color: var(--text-secondary) !important;
    }
    
    /* === INFO BOX === */
    .stAlert {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* === DIVIDER === */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--glass-border) 50%, transparent 100%);
        margin: 20px 0;
    }
    
    /* === THRESHOLD DISPLAY === */
    .threshold-display {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(59, 130, 246, 0.08) 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 16px 0;
    }
    
    .threshold-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        letter-spacing: 1.5px;
        color: rgba(168, 85, 247, 0.8);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .threshold-value {
        font-family: 'Orbitron', monospace;
        font-size: 22px;
        font-weight: 600;
        color: var(--accent-purple);
    }
    
</style>
<script>
    window.parent.playPythonBeep = function(audioBase64) {
        const audio = new Audio(audioBase64);
        audio.play().catch(e => console.log("Audio play blocked: " + e));
    };
</script>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'beat_log' not in st.session_state:
    st.session_state.beat_log = []

def add_to_log(bpm, status, confidence):
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "bpm": bpm,
        "status": status,
        "confidence": f"{confidence:.1%}"
    }
    st.session_state.beat_log.insert(0, entry)  # Prepend latest
    if len(st.session_state.beat_log) > 100:  # Limit history
        st.session_state.beat_log.pop()

# --- ASSET LOADING ---
@st.cache_resource
def load_model():
    path = "models/mitbih_residual_cnn.h5"
    return tf.keras.models.load_model(path) if os.path.exists(path) else None

model = load_model()

# --- ECG LOGIC ---
def generate_sample(t, hr=75, abnormal=False):
    rr = 60 / hr
    phase = (t % rr) / rr
    p = 0.12 * np.exp(-((phase - 0.2)**2) / 0.002)
    w = 0.0004 if not abnormal else 0.003
    q = -0.15 * np.exp(-((phase - 0.44)**2) / w)
    r = 1.0 * np.exp(-((phase - 0.46)**2) / w)
    s = -0.25 * np.exp(-((phase - 0.48)**2) / w)
    t_w = 0.3 * np.exp(-((phase - 0.7)**2) / 0.01)
    return phase, p + q + r + s + t_w + 0.01 * np.random.randn()

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1 class="header-title">
        <span class="header-heartbeat">❤️</span>
        INTELLI<span>HEART</span> PRO
    </h1>
    <p class="header-subtitle">Advanced AI-Powered Cardiac Monitoring System</p>
</div>
""", unsafe_allow_html=True)

# --- DASHBOARD LAYOUT ---
sidebar_col, main_col = st.columns([1.2, 4])

with sidebar_col:
    st.markdown('<div class="config-title">CONTROL PANEL</div>', unsafe_allow_html=True)
    
    mode = st.selectbox("👤 PATIENT PROFILE", ["NORMAL", "HIGH-RISK"], 
                        help="Select patient risk profile for adaptive thresholding")
    
    # Threshold Logic
    if mode == "NORMAL":
        THRESHOLD = 0.45
    else:
        THRESHOLD = 0.2
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="threshold-display">
        <div class="threshold-label">Alarm Threshold</div>
        <div class="threshold-value">{THRESHOLD:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    run = st.toggle("🔴 LIVE MONITOR", value=False, help="Toggle real-time cardiac monitoring")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Metric placeholders
    hr_placeholder = st.empty()
    prob_placeholder = st.empty()
    status_placeholder = st.empty()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Export Section
    if st.button("🗑️ CLEAR HISTORY"):
        st.session_state.beat_log = []
        st.rerun()

    if st.session_state.beat_log:
        import pandas as pd
        df = pd.DataFrame(st.session_state.beat_log)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 EXPORT SESSION",
            csv,
            f"cardiac_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            help="Download all recorded beat data as CSV"
        )

with main_col:
    # ECG Display Header
    st.markdown("""
    <div class="ecg-container">
        <div class="ecg-header">
            <div class="ecg-title">
                📊 LEAD II ECG WAVEFORM
            </div>
            <div class="ecg-live-badge">
                <div class="ecg-live-dot"></div>
                LIVE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    plot_placeholder = st.empty()
    beep_trigger = st.empty()
    
    st.markdown('<div class="config-title">CLINICAL AUDIT TRAIL</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()

# --- MONITORING LOOP ---
if run and model:
    inf_buf = deque(maxlen=187)
    dis_buf = deque([0.0]*600, maxlen=600)
    t, counter = 0, 0
    dt = 1/125
    last_phase, last_prob, last_det = 0, 0, False
    
    # Heart Rate Calculation Logic
    last_peak_time = 0
    calculated_hr = 0

    # Advanced Alarm Buffering
    abnormal_streak = 0

    while run:
        target_hr = 85 if mode == "HIGH-RISK" else 72
        phase, val = generate_sample(t, hr=target_hr, abnormal=(mode == "HIGH-RISK"))
        
        inf_buf.append(val)
        dis_buf.append(val)
        
        # --- R-PEAK DETECTION & PYTHON BEEP ---
        if last_phase < 0.46 <= phase:
            # Calculate HR based on interval
            if last_peak_time > 0:
                interval = t - last_peak_time
                calculated_hr = int(60 / interval)
            last_peak_time = t
            
            # Trigger Python-generated sound
            sound_data = ABNORMAL_BEEP_B64 if last_det else NORMAL_BEEP_B64
            with beep_trigger:
                components.html(f"<script>window.parent.playPythonBeep('{sound_data}');</script>", height=0, width=0)
            
            # Record Beat in Log
            status_text = "⚠️ ABNORMAL" if last_det else "✅ NORMAL"
            add_to_log(calculated_hr, status_text, last_prob)
        
        last_phase = phase
        counter += 1
        
        if len(inf_buf) == 187 and counter % 40 == 0:
            window = np.array(inf_buf)
            window = (window - window.mean()) / (window.std() + 1e-8)
            raw_prob = model.predict(window.reshape(1, 187, 1), verbose=0)[0][0]
            
            # SIGNAL VARIABILITY INDEX (SVI)
            last_prob = 1.0 - raw_prob
            
            # CONSECUTIVE ALARM LOGIC
            if raw_prob > THRESHOLD:
                abnormal_streak += 1
            else:
                abnormal_streak = 0
            
            # Trigger "ABNORMAL" only on 2+ consecutive beats
            last_det = abnormal_streak >= 2

        if counter % 12 == 0:
            # 1. Plot with enhanced styling
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='#05070a')
            ax.set_facecolor('#05070a')
            
            # Grid lines for ECG paper effect
            ax.set_axisbelow(True)
            ax.grid(True, which='major', color='#00ffd0', alpha=0.08, linestyle='-', linewidth=0.5)
            ax.grid(True, which='minor', color='#00ffd0', alpha=0.03, linestyle='-', linewidth=0.3)
            ax.minorticks_on()
            
            # ECG line with glow effect
            ecg_color = "#ff3e3e" if last_det else "#00ffd0"
            ax.plot(list(dis_buf), color=ecg_color, lw=2.5, alpha=0.9)
            ax.plot(list(dis_buf), color=ecg_color, lw=5, alpha=0.15)  # Glow effect
            
            ax.set_ylim(-0.3, 1.4)
            ax.set_xlim(0, 600)
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors='#1e293b', which='both')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            plt.tight_layout(pad=0.5)
            plot_placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)

            # 2. Telemetry with enhanced UI
            hr_class = "metric-value-abnormal" if last_det else "metric-value-normal"
            hr_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💓 Heart Rate (BPM)</div>
                <div class="metric-value {hr_class}">{calculated_hr}</div>
            </div>
            """, unsafe_allow_html=True)
            
            prob_class = "metric-value-abnormal" if last_det else "metric-value-normal"
            prob_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📈 Signal Variability Index</div>
                <div class="metric-value {prob_class}">{last_prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if last_det:
                status_placeholder.markdown("""
                <div class="metric-card">
                    <div class="metric-label">⚡ System Status</div>
                    <div class="status-badge status-abnormal">
                        <span class="status-dot status-dot-abnormal"></span>
                        🚨 ABNORMAL DETECTED
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_placeholder.markdown("""
                <div class="metric-card">
                    <div class="metric-label">⚡ System Status</div>
                    <div class="status-badge status-normal">
                        <span class="status-dot status-dot-normal"></span>
                        ✅ RHYTHM STABLE
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 3. Update Audit Trail Table
            if st.session_state.beat_log:
                import pandas as pd
                log_df = pd.DataFrame(st.session_state.beat_log).head(8) # Show last 8 in UI
                log_placeholder.table(log_df)

        t += dt
        time.sleep(0.005)
else:
    st.markdown("""
    <div class="standby-container">
        <div class="standby-icon">🫀</div>
        <div class="standby-title">SYSTEM STANDBY</div>
        <div class="standby-text">Toggle 'Live Monitor' to begin cardiac analysis</div>
    </div>
    """, unsafe_allow_html=True)
