"""
BrugadaAI — Brugada Syndrome ECG Classification System
Streamlit Application for Clinical Decision Support
"""

import os
import tempfile
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

# ============================================================
# 1. PAGE CONFIG & CSS THEME
# ============================================================

st.set_page_config(
    page_title="BrugadaAI — ECG Classification",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MEDICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0c2d48 0%, #145374 50%, #2e86ab 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}
.main-header h1 {
    margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px;
}
.main-header p {
    margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.85;
}

/* Metric Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 4px solid #145374;
    margin-bottom: 0.8rem;
}
.metric-card.danger { border-left-color: #e74c3c; }
.metric-card.success { border-left-color: #27ae60; }
.metric-card.warning { border-left-color: #f39c12; }

.metric-card h3 {
    margin: 0; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 1px; color: #7f8c8d; font-weight: 600;
}
.metric-card .value {
    font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 0.3rem 0;
}
.metric-card .sub { font-size: 0.8rem; color: #95a5a6; }

/* Result Panels */
.result-panel {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
}
.result-normal {
    background: linear-gradient(135deg, #d5f5e3, #abebc6);
    border: 2px solid #27ae60;
}
.result-brugada {
    background: linear-gradient(135deg, #fadbd8, #f1948a);
    border: 2px solid #e74c3c;
}
.result-panel h2 { margin: 0; font-size: 1.5rem; }
.result-panel p { margin: 0.3rem 0 0 0; font-size: 0.95rem; }

/* Info box */
.info-box {
    background: #eaf2f8;
    border-left: 4px solid #2e86ab;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #2c3e50;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c2d48 0%, #145374 100%);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label span {
    color: #ecf0f1 !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
}

/* Hide default streamlit footer */
footer { visibility: hidden; }
</style>
"""

st.markdown(MEDICAL_CSS, unsafe_allow_html=True)

# ============================================================
# 2. CONSTANTS
# ============================================================

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 100
DURATION_SEC = 12
N_SAMPLES = FS * DURATION_SEC  # 1200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, 'metadata.csv')
FILES_DIR = os.path.join(BASE_DIR, 'files')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ============================================================
# 3. MODEL DEFINITIONS (CNN - must match notebook exactly)
# ============================================================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class ECG_CNN1D(nn.Module):
        def __init__(self, n_leads=12, dropout=0.5):
            super(ECG_CNN1D, self).__init__()
            self.block1 = nn.Sequential(
                nn.Conv1d(n_leads, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(kernel_size=2), nn.Dropout(0.2)
            )
            self.block2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(kernel_size=2), nn.Dropout(0.2)
            )
            self.block3 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.MaxPool1d(kernel_size=2), nn.Dropout(0.2)
            )
            self.block4 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 1), nn.Sigmoid()
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.classifier(x)
            return x.squeeze(1)

# ============================================================
# 4. SIGNAL PROCESSING
# ============================================================

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

def minmax_normalize(signal):
    sig_min = signal.min(axis=0)
    sig_max = signal.max(axis=0)
    denom = sig_max - sig_min
    denom[denom == 0] = 1
    return (signal - sig_min) / denom

# ============================================================
# 5. FEATURE EXTRACTION (must match Classical notebook exactly)
# ============================================================

def detect_rpeaks(signal_1d, fs=100):
    min_distance = int(0.5 * fs)
    height = np.percentile(signal_1d, 70)
    peaks, _ = find_peaks(signal_1d, distance=min_distance, height=height)
    return peaks

def extract_qrs_features(signal_1d, rpeaks, fs=100):
    pre, post = int(0.05 * fs), int(0.08 * fs)
    qrs_durations, qrs_amplitudes, rs_ratios = [], [], []
    for r in rpeaks:
        if r - pre < 0 or r + post >= len(signal_1d):
            continue
        qrs = signal_1d[r - pre: r + post]
        above = np.sum(np.abs(qrs) > 0.2 * np.max(np.abs(qrs)))
        qrs_durations.append(above / fs * 1000)
        qrs_amplitudes.append(signal_1d[r])
        r_val = np.max(qrs)
        s_val = np.abs(np.min(qrs))
        rs_ratios.append(r_val / (s_val + 1e-6))
    if len(qrs_durations) == 0:
        return [0] * 9
    return [
        np.mean(qrs_durations), np.std(qrs_durations),
        np.mean(qrs_amplitudes), np.std(qrs_amplitudes),
        np.max(qrs_amplitudes),
        np.mean(rs_ratios), np.std(rs_ratios),
        np.max(rs_ratios), np.min(rs_ratios)
    ]

def extract_st_features(signal_1d, rpeaks, fs=100):
    st_start, st_end = int(0.08 * fs), int(0.16 * fs)
    st_elevations, st_slopes, st_areas = [], [], []
    for r in rpeaks:
        if r + st_end >= len(signal_1d):
            continue
        st_seg = signal_1d[r + st_start: r + st_end]
        baseline = np.mean(signal_1d[max(0, r - 50):max(1, r - 10)])
        st_elevations.append(np.mean(st_seg) - baseline)
        x = np.arange(len(st_seg))
        st_slopes.append(np.polyfit(x, st_seg, 1)[0])
        st_areas.append(np.trapz(st_seg))
    if len(st_elevations) == 0:
        return [0] * 9
    return [
        np.mean(st_elevations), np.std(st_elevations),
        np.max(st_elevations), np.min(st_elevations),
        np.mean(st_slopes), np.std(st_slopes),
        np.max(st_slopes),
        np.mean(st_areas), np.std(st_areas)
    ]

def extract_twave_features(signal_1d, rpeaks, fs=100):
    t_start, t_end = int(0.16 * fs), int(0.35 * fs)
    t_amplitudes, t_areas, t_inversions = [], [], []
    for r in rpeaks:
        if r + t_end >= len(signal_1d):
            continue
        t_wave = signal_1d[r + t_start: r + t_end]
        t_amplitudes.append(np.max(t_wave) - np.min(t_wave))
        t_areas.append(np.trapz(t_wave))
        t_inversions.append(1 if np.mean(t_wave) < 0 else 0)
    if len(t_amplitudes) == 0:
        return [0] * 5
    return [
        np.mean(t_amplitudes), np.std(t_amplitudes),
        np.mean(t_areas),
        np.mean(t_inversions), np.sum(t_inversions)
    ]

def extract_rr_features(rpeaks, fs=100):
    if len(rpeaks) < 2:
        return [0] * 5
    rr_intervals = np.diff(rpeaks) / fs * 1000
    heart_rate = 60 / (np.mean(rr_intervals) / 1000)
    return [
        np.mean(rr_intervals), np.std(rr_intervals),
        np.min(rr_intervals), np.max(rr_intervals),
        heart_rate
    ]

def extract_pr_features(signal_1d, rpeaks, fs=100):
    p_start, p_end = int(0.20 * fs), int(0.05 * fs)
    pr_intervals, p_amplitudes = [], []
    for r in rpeaks:
        if r - p_start < 0:
            continue
        p_window = signal_1d[r - p_start: r - p_end]
        p_peaks, _ = find_peaks(p_window, height=np.percentile(p_window, 60))
        if len(p_peaks) == 0:
            continue
        p_pos = r - p_start + p_peaks[-1]
        pr_intervals.append((r - p_pos) / fs * 1000)
        p_amplitudes.append(signal_1d[p_pos])
    if len(pr_intervals) == 0:
        return [0] * 5
    return [
        np.mean(pr_intervals), np.std(pr_intervals),
        np.min(pr_intervals), np.max(pr_intervals),
        np.mean(p_amplitudes)
    ]

def extract_qt_features(signal_1d, rpeaks, fs=100):
    qt_end = int(0.45 * fs)
    qt_intervals, qtc_intervals = [], []
    rr_intervals = np.diff(rpeaks) / fs if len(rpeaks) > 1 else [1.0]
    for i, r in enumerate(rpeaks):
        if r + qt_end >= len(signal_1d):
            continue
        qt_window = signal_1d[r: r + qt_end]
        t_region = qt_window[int(0.15 * fs):]
        baseline = np.mean(signal_1d[max(0, r - 30):r])
        cross_idx = np.where(np.diff(np.sign(t_region - baseline)))[0]
        qt = (int(0.15 * fs) + cross_idx[0]) / fs * 1000 if len(cross_idx) > 0 else qt_end / fs * 1000
        qt_intervals.append(qt)
        rr = rr_intervals[min(i, len(rr_intervals) - 1)]
        qtc_intervals.append(qt / np.sqrt(rr))
    if len(qt_intervals) == 0:
        return [0] * 6
    return [
        np.mean(qt_intervals), np.std(qt_intervals),
        np.max(qt_intervals),
        np.mean(qtc_intervals), np.std(qtc_intervals),
        np.max(qtc_intervals)
    ]

def extract_clinical_features(signal, fs=100):
    """Model 1: Statistik + QRS + ST + T-wave + RR → 413 fitur"""
    features = []
    for lead_idx in range(signal.shape[1]):
        x = signal[:, lead_idx]
        features += [
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.max(x) - np.min(x), skew(x), kurtosis(x),
            np.sqrt(np.mean(x ** 2)), np.sum(x ** 2)
        ]
        f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
        features += [f[np.argmax(pxx)], np.sum(pxx)]
    for lead_idx in range(signal.shape[1]):
        x = signal[:, lead_idx]
        rpeaks = detect_rpeaks(x, fs)
        features += extract_qrs_features(x, rpeaks, fs)
        features += extract_st_features(x, rpeaks, fs)
        features += extract_twave_features(x, rpeaks, fs)
    rpeaks_ii = detect_rpeaks(signal[:, 1], fs)
    features += extract_rr_features(rpeaks_ii, fs)
    return np.array(features)

def extract_full_features(signal, fs=100):
    """Model 2: Statistik + QRS + ST + T-wave + PR + QT + RR → 545 fitur"""
    features = []
    for lead_idx in range(signal.shape[1]):
        x = signal[:, lead_idx]
        features += [
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.max(x) - np.min(x), skew(x), kurtosis(x),
            np.sqrt(np.mean(x ** 2)), np.sum(x ** 2)
        ]
        f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
        features += [f[np.argmax(pxx)], np.sum(pxx)]
    for lead_idx in range(signal.shape[1]):
        x = signal[:, lead_idx]
        rpeaks = detect_rpeaks(x, fs)
        features += extract_qrs_features(x, rpeaks, fs)
        features += extract_st_features(x, rpeaks, fs)
        features += extract_twave_features(x, rpeaks, fs)
        features += extract_pr_features(x, rpeaks, fs)
        features += extract_qt_features(x, rpeaks, fs)
    rpeaks_ii = detect_rpeaks(signal[:, 1], fs)
    features += extract_rr_features(rpeaks_ii, fs)
    return np.array(features)

# ============================================================
# 6. MODEL LOADING
# ============================================================

@st.cache_resource
def load_cnn_models():
    """Load CNN model weights from saved file."""
    path = os.path.join(MODELS_DIR, 'cnn_models.pth')
    if not os.path.exists(path) or not TORCH_AVAILABLE:
        return None
    device = torch.device('cpu')
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = ECG_CNN1D(dropout=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model_aug = ECG_CNN1D(dropout=0.5)
    model_aug.load_state_dict(checkpoint['model_aug_state_dict'])
    model_aug.eval()

    return {
        'model': model,
        'model_aug': model_aug,
        'threshold': checkpoint['threshold'],
        'threshold_aug': checkpoint['threshold_aug'],
        'device': device,
    }

@st.cache_resource
def load_xgboost_models():
    """Load XGBoost models from saved file."""
    path = os.path.join(MODELS_DIR, 'xgboost_models.pkl')
    if not os.path.exists(path):
        return None
    try:
        import joblib
        data = joblib.load(path)
        return data
    except Exception:
        return None

@st.cache_data
def load_metadata():
    if os.path.exists(META_PATH):
        df = pd.read_csv(META_PATH)
        df['label'] = (df['brugada'] > 0).astype(int)
        return df
    return None

def load_ecg_wfdb(patient_id):
    """Load ECG from dataset using wfdb."""
    import wfdb
    path = os.path.join(FILES_DIR, str(patient_id), str(patient_id))
    record = wfdb.rdrecord(path)
    return record.p_signal, record.sig_name, record.fs

def load_ecg_from_upload(hea_bytes, dat_bytes, filename_base):
    """Load ECG from uploaded .hea and .dat files."""
    import wfdb
    with tempfile.TemporaryDirectory() as tmp_dir:
        hea_path = os.path.join(tmp_dir, filename_base + '.hea')
        dat_path = os.path.join(tmp_dir, filename_base + '.dat')
        with open(hea_path, 'wb') as f:
            f.write(hea_bytes)
        with open(dat_path, 'wb') as f:
            f.write(dat_bytes)
        record = wfdb.rdrecord(os.path.join(tmp_dir, filename_base))
        return record.p_signal, record.sig_name, record.fs

# ============================================================
# 7. PREDICTION FUNCTIONS
# ============================================================

def predict_cnn(signal_norm, cnn_data, use_aug=False):
    """Run CNN prediction on normalized signal."""
    if cnn_data is None or not TORCH_AVAILABLE:
        return None
    key = 'model_aug' if use_aug else 'model'
    thr_key = 'threshold_aug' if use_aug else 'threshold'
    model = cnn_data[key]
    threshold = cnn_data[thr_key]
    device = cnn_data['device']

    x = torch.tensor(
        signal_norm.T[np.newaxis, :, :], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        output = model(x)
        # Double sigmoid to match training evaluation
        prob = torch.sigmoid(output).cpu().numpy()[0]

    return {
        'probability': float(prob),
        'threshold': float(threshold),
        'prediction': int(prob >= threshold),
        'label': 'Brugada' if prob >= threshold else 'Normal',
    }

def predict_xgboost(signal_filtered, xgb_data, model_key='m1'):
    """Run XGBoost prediction."""
    if xgb_data is None:
        return None
    if model_key == 'm1':
        features = extract_clinical_features(signal_filtered)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = xgb_data['scaler_c'].transform(features.reshape(1, -1))
        prob = xgb_data['xgb_m1'].predict_proba(features_scaled)[0, 1]
        threshold = xgb_data['threshold_m1']
    else:
        features = extract_full_features(signal_filtered)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = xgb_data['scaler_f'].transform(features.reshape(1, -1))
        prob = xgb_data['xgb_m2'].predict_proba(features_scaled)[0, 1]
        threshold = xgb_data['threshold_m2']

    return {
        'probability': float(prob),
        'threshold': float(threshold),
        'prediction': int(prob >= threshold),
        'label': 'Brugada' if prob >= threshold else 'Normal',
    }

# ============================================================
# 8. VISUALIZATION FUNCTIONS
# ============================================================

ECG_COLORS = [
    '#e74c3c', '#2980b9', '#27ae60', '#8e44ad',
    '#f39c12', '#1abc9c', '#d35400', '#2c3e50',
    '#c0392b', '#3498db', '#2ecc71', '#9b59b6',
]

def plot_12_lead_ecg(signal, lead_names, fs=100, title="12-Lead ECG"):
    """Interactive ECG paper-style 12-lead plot."""
    duration = signal.shape[0] / fs
    time = np.arange(signal.shape[0]) / fs

    fig = make_subplots(
        rows=6, cols=2,
        subplot_titles=[f"Lead {n}" for n in lead_names],
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    # Standard order: Column 1 = I, II, III, aVR, aVL, aVF; Column 2 = V1-V6
    col1_idx = [0, 1, 2, 3, 4, 5]  # I, II, III, aVR, aVL, aVF
    col2_idx = [6, 7, 8, 9, 10, 11]  # V1-V6

    for row, idx in enumerate(col1_idx):
        fig.add_trace(
            go.Scatter(
                x=time, y=signal[:, idx],
                line=dict(color=ECG_COLORS[idx], width=1.2),
                name=lead_names[idx],
                hovertemplate=f"Lead {lead_names[idx]}<br>Time: %{{x:.2f}}s<br>Amp: %{{y:.4f}}mV<extra></extra>",
            ),
            row=row + 1, col=1,
        )

    for row, idx in enumerate(col2_idx):
        fig.add_trace(
            go.Scatter(
                x=time, y=signal[:, idx],
                line=dict(color=ECG_COLORS[idx], width=1.2),
                name=lead_names[idx],
                hovertemplate=f"Lead {lead_names[idx]}<br>Time: %{{x:.2f}}s<br>Amp: %{{y:.4f}}mV<extra></extra>",
            ),
            row=row + 1, col=2,
        )

    fig.update_layout(
        height=900,
        title=dict(text=title, font=dict(size=16, color='#2c3e50')),
        showlegend=False,
        plot_bgcolor='#fff5f5',
        paper_bgcolor='white',
        font=dict(family='Inter', size=10),
        margin=dict(l=50, r=20, t=60, b=30),
    )

    for i in range(1, 13):
        r = ((i - 1) % 6) + 1
        c = 1 if i <= 6 else 2
        fig.update_xaxes(
            gridcolor='rgba(255,150,150,0.3)',
            gridwidth=1,
            dtick=0.2,
            minor=dict(gridcolor='rgba(255,150,150,0.12)', gridwidth=0.5, dtick=0.04),
            range=[0, duration],
            row=r, col=c,
        )
        fig.update_yaxes(
            gridcolor='rgba(255,150,150,0.3)',
            gridwidth=1,
            row=r, col=c,
        )

    return fig

def plot_single_lead(signal_1d, lead_name, fs=100, rpeaks=None):
    """Detailed single lead view with R-peak markers."""
    time = np.arange(len(signal_1d)) / fs
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time, y=signal_1d,
        line=dict(color='#2980b9', width=1.5),
        name=f'Lead {lead_name}',
        hovertemplate="Time: %{x:.2f}s<br>Amp: %{y:.4f}mV<extra></extra>",
    ))

    if rpeaks is not None and len(rpeaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[rpeaks], y=signal_1d[rpeaks],
            mode='markers',
            marker=dict(color='#e74c3c', size=8, symbol='triangle-up'),
            name='R-Peaks',
            hovertemplate="R-Peak<br>Time: %{x:.2f}s<br>Amp: %{y:.4f}mV<extra></extra>",
        ))

    fig.update_layout(
        height=300,
        title=f'Lead {lead_name} — Detail View',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV)',
        plot_bgcolor='#fff5f5',
        paper_bgcolor='white',
        font=dict(family='Inter'),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor='rgba(255,150,150,0.3)', dtick=0.2),
        yaxis=dict(gridcolor='rgba(255,150,150,0.3)'),
    )
    return fig

def plot_risk_gauge(probability, threshold):
    """Radial gauge chart for risk assessment."""
    color = '#27ae60' if probability < threshold else '#e74c3c'
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number=dict(suffix='%', font=dict(size=36)),
        delta=dict(
            reference=threshold * 100,
            increasing=dict(color='#e74c3c'),
            decreasing=dict(color='#27ae60'),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=2, tickcolor='#2c3e50'),
            bar=dict(color=color),
            bgcolor='white',
            borderwidth=2,
            bordercolor='#bdc3c7',
            steps=[
                dict(range=[0, 30], color='#d5f5e3'),
                dict(range=[30, 60], color='#fdebd0'),
                dict(range=[60, 100], color='#fadbd8'),
            ],
            threshold=dict(
                line=dict(color='#2c3e50', width=3),
                thickness=0.8,
                value=threshold * 100,
            ),
        ),
        title=dict(text='Brugada Probability', font=dict(size=14)),
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor='white',
        font=dict(family='Inter'),
    )
    return fig

def plot_model_comparison(results):
    """Bar chart comparing model predictions."""
    models = list(results.keys())
    probs = [results[m]['probability'] * 100 for m in models]
    thresholds = [results[m]['threshold'] * 100 for m in models]
    colors = ['#e74c3c' if results[m]['prediction'] == 1 else '#27ae60' for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=probs,
        marker_color=colors,
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside',
        name='Probability',
    ))
    for i, (m, t) in enumerate(zip(models, thresholds)):
        fig.add_shape(
            type='line',
            x0=i - 0.4, x1=i + 0.4, y0=t, y1=t,
            line=dict(color='#2c3e50', width=2, dash='dash'),
        )

    fig.update_layout(
        height=350,
        title='Model Comparison — Brugada Probability',
        yaxis_title='Probability (%)',
        yaxis_range=[0, max(max(probs) + 15, 100)],
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter'),
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig

# ============================================================
# 9. REPORT GENERATOR
# ============================================================

def generate_report(patient_id, results, signal_info):
    """Generate a downloadable clinical text report."""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [
        "=" * 58,
        "     BRUGADA SYNDROME SCREENING REPORT",
        "     BrugadaAI — Clinical Decision Support",
        "=" * 58,
        "",
        f"  Report Date     : {now}",
        f"  Patient ID      : {patient_id}",
        f"  Sampling Rate   : {signal_info.get('fs', 100)} Hz",
        f"  Duration        : {signal_info.get('duration', 12)} seconds",
        f"  Leads           : 12 standard ECG leads",
        "",
        "-" * 58,
        "  MODEL PREDICTIONS",
        "-" * 58,
    ]

    brugada_count = 0
    for name, res in results.items():
        pred_label = res['label']
        if res['prediction'] == 1:
            brugada_count += 1
        lines += [
            "",
            f"  {name}",
            f"    Probability    : {res['probability']:.4f} ({res['probability']*100:.1f}%)",
            f"    Threshold      : {res['threshold']:.2f}",
            f"    Classification : {pred_label}",
        ]

    total = len(results)
    if total > 0:
        consensus = brugada_count / total
        if consensus == 0:
            risk = "RENDAH (LOW)"
        elif consensus < 0.5:
            risk = "SEDANG (MODERATE)"
        elif consensus < 1.0:
            risk = "TINGGI (HIGH)"
        else:
            risk = "SANGAT TINGGI (VERY HIGH)"
    else:
        risk = "N/A"

    lines += [
        "",
        "-" * 58,
        "  ENSEMBLE RESULT",
        "-" * 58,
        f"  Consensus      : {brugada_count}/{total} models indicate Brugada",
        f"  Risk Level     : {risk}",
        "",
        "-" * 58,
        "  DISCLAIMER",
        "-" * 58,
        "  Hasil ini merupakan alat skrining berbasis AI dan BUKAN",
        "  diagnosis medis definitif. Hasil harus dikonfirmasi oleh",
        "  dokter spesialis jantung (kardiolog) yang berkualifikasi.",
        "",
        "  This is an AI-based screening tool and NOT a definitive",
        "  medical diagnosis. Results should be confirmed by a",
        "  qualified cardiologist.",
        "",
        "=" * 58,
    ]
    return "\n".join(lines)

# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🫀 BrugadaAI")
    st.markdown("*ECG Classification System*")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠 Dashboard", "🔬 Analisis ECG", "📂 Dataset Explorer",
         "📊 Perbandingan Model", "ℹ️ Tentang"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status
    cnn_data = load_cnn_models()
    xgb_data = load_xgboost_models()

    st.markdown("### 📦 Status Model")
    if cnn_data:
        st.success("CNN 1D: Loaded ✓")
        st.success("CNN 1D + Aug: Loaded ✓")
    else:
        st.warning("CNN Models: Not found")

    if xgb_data:
        st.success("XGBoost Clinical: Loaded ✓")
        st.success("XGBoost Full: Loaded ✓")
    else:
        st.warning("XGBoost Models: Not found")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.75rem; opacity:0.6;'>"
        "BrugadaAI v1.0<br>IIDSC 2026</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# 11. PAGE: DASHBOARD
# ============================================================

def page_dashboard():
    st.markdown(
        '<div class="main-header">'
        '<h1>🫀 BrugadaAI Detection System</h1>'
        '<p>Sistem Klasifikasi Sindrom Brugada Berbasis Kecerdasan Buatan</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    metadata = load_metadata()

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    total = len(metadata) if metadata is not None else 0
    normal = int((metadata['label'] == 0).sum()) if metadata is not None else 0
    brugada = int((metadata['label'] == 1).sum()) if metadata is not None else 0
    models_loaded = sum([cnn_data is not None, xgb_data is not None]) * 2

    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<h3>Total Pasien</h3><div class="value">{total}</div>'
            f'<div class="sub">Dataset Brugada-HUCA</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card success">'
            f'<h3>Normal</h3><div class="value">{normal}</div>'
            f'<div class="sub">{normal/max(total,1)*100:.1f}% dari total</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card danger">'
            f'<h3>Brugada</h3><div class="value">{brugada}</div>'
            f'<div class="sub">{brugada/max(total,1)*100:.1f}% dari total</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card warning">'
            f'<h3>Model Aktif</h3><div class="value">{models_loaded}/4</div>'
            f'<div class="sub">Siap untuk prediksi</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("### 📋 Tentang Sistem")
        st.markdown(
            '<div class="info-box">'
            '<b>BrugadaAI</b> adalah sistem skrining Sindrom Brugada berbasis AI yang '
            'menganalisis rekaman EKG 12-Lead untuk mendeteksi pola karakteristik ST-elevasi '
            'pada lead V1–V3. Sistem ini menggunakan kombinasi model deep learning (CNN) '
            'dan classical machine learning (XGBoost) untuk memberikan prediksi yang robust.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 🤖 Model yang Tersedia")
        model_info = pd.DataFrame({
            'Model': ['CNN 1D', 'CNN 1D + Augmentasi', 'XGBoost Clinical', 'XGBoost Full'],
            'Tipe': ['Deep Learning', 'Deep Learning', 'Classical ML', 'Classical ML'],
            'Fitur': ['Raw ECG', 'Raw ECG (Aug)', '413 Features', '545 Features'],
            'Keunggulan': ['Best overall', 'Robust', 'Best F1', 'Best Recall'],
        })
        st.dataframe(model_info, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### 📊 Distribusi Dataset")
        if metadata is not None:
            fig = go.Figure(go.Pie(
                labels=['Normal', 'Brugada'],
                values=[normal, brugada],
                marker=dict(colors=['#27ae60', '#e74c3c']),
                hole=0.5,
                textinfo='label+value+percent',
                textfont=dict(size=14),
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='white',
                font=dict(family='Inter'),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ⚡ Parameter EKG")
        st.markdown(
            "- **Sampling Rate:** 100 Hz\n"
            "- **Durasi:** 12 detik (1200 sampel)\n"
            "- **Lead:** 12 lead standar\n"
            "- **Format:** WFDB (.dat + .hea)"
        )

    # Quick-start guide
    if not cnn_data and not xgb_data:
        st.markdown("---")
        st.warning(
            "⚠️ **Belum ada model yang dimuat.** Jalankan `export_models.py` "
            "di notebook Anda terlebih dahulu untuk menyimpan model terlatih, "
            "lalu restart aplikasi ini."
        )

# ============================================================
# 12. PAGE: ANALISIS ECG
# ============================================================

def page_analisis_ecg():
    st.markdown(
        '<div class="main-header">'
        '<h1>🔬 Analisis ECG</h1>'
        '<p>Upload atau pilih data EKG untuk klasifikasi Sindrom Brugada</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    metadata = load_metadata()

    # Input method
    tab_upload, tab_dataset = st.tabs(["📤 Upload File WFDB", "📂 Pilih dari Dataset"])

    signal = None
    sig_names = LEAD_NAMES
    fs_loaded = FS
    patient_label = "Upload"
    ground_truth = None

    with tab_upload:
        st.markdown(
            '<div class="info-box">'
            'Upload file <b>.hea</b> dan <b>.dat</b> dari rekaman EKG WFDB. '
            'Kedua file harus memiliki nama yang sama (contoh: 188981.hea & 188981.dat).'
            '</div>',
            unsafe_allow_html=True,
        )
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            hea_file = st.file_uploader("Upload .hea file", type=['hea'], key='hea')
        with col_u2:
            dat_file = st.file_uploader("Upload .dat file", type=['dat'], key='dat')

        if hea_file and dat_file:
            basename = os.path.splitext(hea_file.name)[0]
            try:
                signal, sig_names, fs_loaded = load_ecg_from_upload(
                    hea_file.read(), dat_file.read(), basename
                )
                patient_label = basename
                st.success(f"✓ File berhasil dimuat: {basename} ({signal.shape[0]} sampel, {len(sig_names)} lead)")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    with tab_dataset:
        if metadata is not None:
            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                patient_ids = metadata['patient_id'].tolist()
                selected_id = st.selectbox(
                    "Pilih Patient ID",
                    patient_ids,
                    format_func=lambda x: f"{x} — {'Brugada' if metadata.loc[metadata['patient_id']==x, 'label'].values[0]==1 else 'Normal'}",
                )
            with col_sel2:
                st.markdown("<br>", unsafe_allow_html=True)
                load_btn = st.button("🔍 Muat Data EKG", use_container_width=True)

            if load_btn or st.session_state.get('last_patient') == selected_id:
                st.session_state['last_patient'] = selected_id
                try:
                    signal, sig_names, fs_loaded = load_ecg_wfdb(selected_id)
                    patient_label = str(selected_id)
                    row = metadata[metadata['patient_id'] == selected_id].iloc[0]
                    ground_truth = int(row['label'])
                    gt_text = "Brugada" if ground_truth == 1 else "Normal"
                    st.success(f"✓ Data dimuat: Patient {selected_id} | Ground Truth: **{gt_text}** | {signal.shape[0]} sampel")
                except Exception as e:
                    st.error(f"Gagal membaca data: {e}")
        else:
            st.info("metadata.csv tidak ditemukan. Gunakan tab Upload.")

    if signal is None:
        st.info("👆 Silakan upload file EKG atau pilih pasien dari dataset untuk memulai analisis.")
        return

    # ---- Signal Processing ----
    st.markdown("---")
    st.markdown("### 📈 Sinyal EKG 12-Lead")

    signal_filtered = bandpass_filter(signal, fs=fs_loaded)
    signal_norm = minmax_normalize(signal_filtered)

    # Display options
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        display_mode = st.radio("Tampilan", ["Filtered", "Raw", "Normalized"], horizontal=True)
    with col_opt2:
        selected_leads = st.multiselect(
            "Lead Detail",
            LEAD_NAMES[:len(sig_names)],
            default=['V1', 'V2', 'V3'],
        )
    with col_opt3:
        show_rpeaks = st.checkbox("Tampilkan R-Peaks", value=True)

    display_signal = {
        'Raw': signal,
        'Filtered': signal_filtered,
        'Normalized': signal_norm,
    }[display_mode]

    # 12-lead plot
    fig_12 = plot_12_lead_ecg(
        display_signal, sig_names[:12], fs=fs_loaded,
        title=f"12-Lead ECG — Patient {patient_label} ({display_mode})"
    )
    st.plotly_chart(fig_12, use_container_width=True)

    # Detail lead views
    if selected_leads:
        st.markdown("### 🔎 Detail Lead")
        for lead_name in selected_leads:
            if lead_name in sig_names:
                idx = sig_names.index(lead_name)
                rpeaks = detect_rpeaks(signal_filtered[:, idx], fs_loaded) if show_rpeaks else None
                fig_lead = plot_single_lead(
                    display_signal[:, idx], lead_name, fs=fs_loaded, rpeaks=rpeaks
                )
                st.plotly_chart(fig_lead, use_container_width=True)

    # ---- ECG Parameters ----
    st.markdown("---")
    st.markdown("### ❤️ Parameter EKG")

    rpeaks_ii = detect_rpeaks(signal_filtered[:, 1], fs_loaded)
    rr_feats = extract_rr_features(rpeaks_ii, fs_loaded)

    cp1, cp2, cp3, cp4, cp5 = st.columns(5)
    with cp1:
        st.metric("Heart Rate", f"{rr_feats[4]:.0f} bpm")
    with cp2:
        st.metric("RR Mean", f"{rr_feats[0]:.0f} ms")
    with cp3:
        st.metric("RR Std", f"{rr_feats[1]:.1f} ms")
    with cp4:
        st.metric("RR Min", f"{rr_feats[2]:.0f} ms")
    with cp5:
        st.metric("RR Max", f"{rr_feats[3]:.0f} ms")

    # ---- PREDICTION ----
    st.markdown("---")
    st.markdown("### 🧠 Hasil Klasifikasi")

    # Threshold controls
    with st.expander("⚙️ Pengaturan Threshold (Opsional)", expanded=False):
        st.markdown("Sesuaikan threshold untuk mengontrol sensitivitas/spesifisitas.")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            custom_thr_cnn = st.slider(
                "CNN Threshold",
                0.0, 1.0,
                value=cnn_data['threshold'] if cnn_data else 0.5,
                step=0.05, key='thr_cnn',
            )
        with tc2:
            custom_thr_cnn_aug = st.slider(
                "CNN+Aug Threshold",
                0.0, 1.0,
                value=cnn_data['threshold_aug'] if cnn_data else 0.5,
                step=0.05, key='thr_cnn_aug',
            )
        with tc3:
            custom_thr_xgb1 = st.slider(
                "XGBoost Clinical Thr",
                0.0, 1.0,
                value=xgb_data['threshold_m1'] if xgb_data else 0.5,
                step=0.05, key='thr_xgb1',
            )
        with tc4:
            custom_thr_xgb2 = st.slider(
                "XGBoost Full Thr",
                0.0, 1.0,
                value=xgb_data['threshold_m2'] if xgb_data else 0.5,
                step=0.05, key='thr_xgb2',
            )

    # Run predictions
    results = {}

    if cnn_data:
        pred = predict_cnn(signal_norm, cnn_data, use_aug=False)
        if pred:
            pred['threshold'] = custom_thr_cnn
            pred['prediction'] = int(pred['probability'] >= custom_thr_cnn)
            pred['label'] = 'Brugada' if pred['prediction'] == 1 else 'Normal'
            results['CNN 1D'] = pred

        pred_aug = predict_cnn(signal_norm, cnn_data, use_aug=True)
        if pred_aug:
            pred_aug['threshold'] = custom_thr_cnn_aug
            pred_aug['prediction'] = int(pred_aug['probability'] >= custom_thr_cnn_aug)
            pred_aug['label'] = 'Brugada' if pred_aug['prediction'] == 1 else 'Normal'
            results['CNN 1D + Aug'] = pred_aug

    if xgb_data:
        pred_m1 = predict_xgboost(signal_filtered, xgb_data, model_key='m1')
        if pred_m1:
            pred_m1['threshold'] = custom_thr_xgb1
            pred_m1['prediction'] = int(pred_m1['probability'] >= custom_thr_xgb1)
            pred_m1['label'] = 'Brugada' if pred_m1['prediction'] == 1 else 'Normal'
            results['XGBoost Clinical'] = pred_m1

        pred_m2 = predict_xgboost(signal_filtered, xgb_data, model_key='m2')
        if pred_m2:
            pred_m2['threshold'] = custom_thr_xgb2
            pred_m2['prediction'] = int(pred_m2['probability'] >= custom_thr_xgb2)
            pred_m2['label'] = 'Brugada' if pred_m2['prediction'] == 1 else 'Normal'
            results['XGBoost Full'] = pred_m2

    if not results:
        st.warning("Tidak ada model yang tersedia. Silakan export model terlebih dahulu.")
        return

    # Display results
    cols = st.columns(len(results))
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            css_class = 'result-brugada' if res['prediction'] == 1 else 'result-normal'
            icon = '⚠️' if res['prediction'] == 1 else '✅'
            st.markdown(
                f'<div class="result-panel {css_class}">'
                f'<h2>{icon} {res["label"]}</h2>'
                f'<p><b>{name}</b></p>'
                f'<p>Probabilitas: {res["probability"]*100:.1f}%</p>'
                f'<p>Threshold: {res["threshold"]:.2f}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Ensemble result
    st.markdown("")
    brugada_count = sum(1 for r in results.values() if r['prediction'] == 1)
    total_models = len(results)

    if brugada_count == 0:
        ensemble_class = "result-normal"
        ensemble_text = "NORMAL — Tidak ada indikasi Brugada"
        ensemble_icon = "✅"
    elif brugada_count < total_models:
        ensemble_class = "result-brugada"
        ensemble_text = f"WASPADA — {brugada_count}/{total_models} model mendeteksi Brugada"
        ensemble_icon = "⚠️"
    else:
        ensemble_class = "result-brugada"
        ensemble_text = f"POSITIF — Semua model ({total_models}/{total_models}) mendeteksi Brugada"
        ensemble_icon = "🚨"

    st.markdown(
        f'<div class="result-panel {ensemble_class}" style="margin: 1rem 0;">'
        f'<h2>{ensemble_icon} ENSEMBLE: {ensemble_text}</h2>'
        f'<p>Konsensus {brugada_count} dari {total_models} model</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if ground_truth is not None:
        gt_txt = "Brugada" if ground_truth == 1 else "Normal"
        st.info(f"📌 **Ground Truth:** {gt_txt}")

    # Gauges & Comparison Chart
    st.markdown("---")
    st.markdown("### 📊 Visualisasi Prediksi")

    tab_gauge, tab_bar = st.tabs(["🎯 Risk Gauge", "📊 Comparison"])

    with tab_gauge:
        gauge_cols = st.columns(len(results))
        for i, (name, res) in enumerate(results.items()):
            with gauge_cols[i]:
                fig_g = plot_risk_gauge(res['probability'], res['threshold'])
                fig_g.update_layout(title=dict(text=name, font=dict(size=12)))
                st.plotly_chart(fig_g, use_container_width=True)

    with tab_bar:
        fig_comp = plot_model_comparison(results)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ---- Report Generation ----
    st.markdown("---")
    st.markdown("### 📄 Laporan Klinis")

    report_text = generate_report(
        patient_label, results,
        {'fs': fs_loaded, 'duration': signal.shape[0] / fs_loaded}
    )

    with st.expander("👁️ Preview Laporan", expanded=False):
        st.code(report_text, language='text')

    st.download_button(
        "⬇️ Download Laporan (.txt)",
        data=report_text,
        file_name=f"BrugadaAI_Report_{patient_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ============================================================
# 13. PAGE: DATASET EXPLORER
# ============================================================

def page_dataset_explorer():
    st.markdown(
        '<div class="main-header">'
        '<h1>📂 Dataset Explorer</h1>'
        '<p>Jelajahi dan analisis dataset Brugada-HUCA</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    metadata = load_metadata()
    if metadata is None:
        st.warning("metadata.csv tidak ditemukan.")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        class_filter = st.selectbox("Filter Kelas", ["Semua", "Normal", "Brugada"])
    with col_f2:
        basal_filter = st.selectbox("Basal Pattern", ["Semua", "Ya (1)", "Tidak (0)"])
    with col_f3:
        death_filter = st.selectbox("Sudden Death", ["Semua", "Ya (1)", "Tidak (0)"])

    df = metadata.copy()
    if class_filter == "Normal":
        df = df[df['label'] == 0]
    elif class_filter == "Brugada":
        df = df[df['label'] == 1]

    if basal_filter == "Ya (1)":
        df = df[df['basal_pattern'] == 1]
    elif basal_filter == "Tidak (0)":
        df = df[df['basal_pattern'] == 0]

    if death_filter == "Ya (1)":
        df = df[df['sudden_death'] == 1]
    elif death_filter == "Tidak (0)":
        df = df[df['sudden_death'] == 0]

    st.markdown(f"**{len(df)}** pasien ditemukan")

    # Display metadata
    display_df = df.copy()
    display_df['status'] = display_df['label'].map({0: '🟢 Normal', 1: '🔴 Brugada'})
    st.dataframe(
        display_df[['patient_id', 'status', 'basal_pattern', 'sudden_death', 'brugada']],
        use_container_width=True,
        height=400,
    )

    # Visualizations
    st.markdown("---")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### 📊 Distribusi Kelas")
        class_counts = df['label'].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=['Normal', 'Brugada'],
            y=[class_counts.get(0, 0), class_counts.get(1, 0)],
            marker_color=['#27ae60', '#e74c3c'],
            text=[class_counts.get(0, 0), class_counts.get(1, 0)],
            textposition='outside',
        ))
        fig.update_layout(
            height=300, paper_bgcolor='white', plot_bgcolor='white',
            yaxis_title='Jumlah Pasien',
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_v2:
        st.markdown("#### 📊 Brugada Grade Distribution")
        grade_counts = df['brugada'].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=[str(g) for g in grade_counts.index],
            y=grade_counts.values,
            marker_color=['#27ae60', '#e74c3c', '#f39c12'],
            text=grade_counts.values,
            textposition='outside',
        ))
        fig.update_layout(
            height=300, paper_bgcolor='white', plot_bgcolor='white',
            xaxis_title='Grade Brugada', yaxis_title='Jumlah',
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick ECG preview
    st.markdown("---")
    st.markdown("#### 👁️ Quick ECG Preview")
    preview_id = st.selectbox(
        "Pilih pasien untuk preview cepat",
        df['patient_id'].tolist(),
        key='preview_select'
    )
    if st.button("📈 Tampilkan EKG", key='show_preview'):
        try:
            sig, names, fs_val = load_ecg_wfdb(preview_id)
            sig_f = bandpass_filter(sig, fs=fs_val)
            fig = plot_12_lead_ecg(
                sig_f, names[:12], fs=fs_val,
                title=f"ECG Preview — Patient {preview_id}"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    # Batch prediction
    st.markdown("---")
    st.markdown("#### 🔄 Batch Prediction")
    st.markdown("Jalankan prediksi pada beberapa pasien sekaligus.")

    n_patients = st.slider("Jumlah pasien", 5, min(50, len(df)), 10)
    if st.button("🚀 Mulai Batch Prediction", key='batch_btn'):
        if not cnn_data and not xgb_data:
            st.warning("Tidak ada model yang tersedia.")
            return

        batch_df = df.head(n_patients).copy()
        batch_results = []
        progress = st.progress(0, "Memproses...")

        for i, (_, row) in enumerate(batch_df.iterrows()):
            try:
                sig, _, fs_val = load_ecg_wfdb(row['patient_id'])
                sig_f = bandpass_filter(sig, fs=fs_val)
                sig_n = minmax_normalize(sig_f)

                res_row = {'patient_id': row['patient_id'], 'ground_truth': row['label']}

                if cnn_data:
                    pred = predict_cnn(sig_n, cnn_data, use_aug=False)
                    if pred:
                        res_row['CNN_prob'] = pred['probability']
                        res_row['CNN_pred'] = pred['label']

                if xgb_data:
                    pred_m1 = predict_xgboost(sig_f, xgb_data, model_key='m1')
                    if pred_m1:
                        res_row['XGB_Clinical_prob'] = pred_m1['probability']
                        res_row['XGB_Clinical_pred'] = pred_m1['label']

                    pred_m2 = predict_xgboost(sig_f, xgb_data, model_key='m2')
                    if pred_m2:
                        res_row['XGB_Full_prob'] = pred_m2['probability']
                        res_row['XGB_Full_pred'] = pred_m2['label']

                batch_results.append(res_row)
            except Exception:
                batch_results.append({'patient_id': row['patient_id'], 'error': True})

            progress.progress((i + 1) / n_patients, f"Memproses {i+1}/{n_patients}...")

        progress.empty()
        result_df = pd.DataFrame(batch_results)
        st.dataframe(result_df, use_container_width=True)

        csv_data = result_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Hasil Batch (.csv)",
            data=csv_data,
            file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# ============================================================
# 14. PAGE: PERBANDINGAN MODEL
# ============================================================

def page_model_comparison():
    st.markdown(
        '<div class="main-header">'
        '<h1>📊 Perbandingan Model</h1>'
        '<p>Analisis performa dan karakteristik setiap model</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Model specifications table
    st.markdown("### 📋 Spesifikasi Model")

    specs = pd.DataFrame({
        'Model': ['CNN 1D', 'CNN 1D + Augmentasi', 'XGBoost Clinical', 'XGBoost Full'],
        'Tipe': ['Deep Learning', 'Deep Learning', 'Gradient Boosting', 'Gradient Boosting'],
        'Input': ['12×1200 sinyal', '12×1200 sinyal', '413 fitur', '545 fitur'],
        'Fitur': [
            'Raw ECG signal',
            'Raw ECG + augmentasi',
            'Stat + QRS + ST + T-wave + RR',
            'Stat + QRS + ST + T-wave + PR + QT + RR',
        ],
        'Parameter': ['~500K', '~500K', 'Grid-tuned', 'Grid-tuned'],
        'Keunggulan': [
            'Learns features automatically',
            'More robust, handles noise',
            'Best F1 score',
            'Best Recall (sensitivity)',
        ],
    })
    st.dataframe(specs, use_container_width=True, hide_index=True)

    # Architecture details
    st.markdown("---")
    tab_cnn, tab_xgb = st.tabs(["🧠 CNN Architecture", "🌲 XGBoost Pipeline"])

    with tab_cnn:
        st.markdown("""
        #### CNN 1D Architecture — ECG_CNN1D

        ```
        Input: (batch, 12, 1200) — 12 leads × 1200 samples

        Block 1: Conv1d(12→64, k=7) → BN → ReLU → MaxPool(2) → Dropout(0.2)
                 Output: (batch, 64, 600)

        Block 2: Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.2)
                 Output: (batch, 128, 300)

        Block 3: Conv1d(128→256, k=3) → BN → ReLU → MaxPool(2) → Dropout(0.2)
                 Output: (batch, 256, 150)

        Block 4: Conv1d(256→256, k=3) → BN → ReLU → AdaptiveAvgPool1d(1)
                 Output: (batch, 256, 1)

        Classifier: Flatten → Linear(256,128) → ReLU → Dropout(0.5) → Linear(128,1) → Sigmoid
                    Output: (batch,) — probability [0, 1]
        ```

        **Preprocessing Pipeline:**
        1. Bandpass filter (0.5-40 Hz, Butterworth order 4)
        2. Min-Max normalization per lead
        3. Transpose to (channels, length) format
        """)

        if TORCH_AVAILABLE:
            model_test = ECG_CNN1D()
            total_params = sum(p.numel() for p in model_test.parameters())
            trainable = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
            st.markdown(f"**Total Parameters:** {total_params:,}")
            st.markdown(f"**Trainable Parameters:** {trainable:,}")

    with tab_xgb:
        st.markdown("""
        #### XGBoost Feature Pipeline

        **Model 1 — Clinical Features (413 fitur):**
        - Statistik per lead (11 × 12 = 132): mean, std, min, max, range, skew, kurtosis, RMS, energy, dom_freq, total_power
        - QRS features per lead (9 × 12 = 108): duration, amplitude, R/S ratio stats
        - ST features per lead (9 × 12 = 108): elevation, slope, area stats
        - T-wave features per lead (5 × 12 = 60): amplitude, area, inversion stats
        - RR interval features (5): mean, std, min, max, heart rate

        **Model 2 — Full Features (545 fitur):**
        - Semua fitur Model 1 (413)
        - PR interval per lead (5 × 12 = 60): interval stats, P-wave amplitude
        - QT interval per lead (6 × 12 = 72): QT and QTc stats

        **Hyperparameter Tuning:** GridSearchCV with Stratified 5-Fold CV
        """)

    # Performance comparison
    st.markdown("---")
    st.markdown("### 📈 Performa Model (dari Training)")

    perf = pd.DataFrame({
        'Model': ['XGBoost Clinical', 'XGBoost Full', 'CNN 1D', 'CNN 1D + Aug'],
        'K-Fold F1': [0.6627, 0.6047, 0.7182, 'See notebook'],
        'K-Fold Recall': [0.7368, 0.8553, 0.7883, 'See notebook'],
        'K-Fold AUC': [0.8806, 0.8802, 0.8635, 'See notebook'],
        'Test F1': [0.7429, 0.6512, 0.8966, 'See notebook'],
        'Test AUC': [0.9034, 0.9069, 0.9351, 'See notebook'],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    # Visualization
    models_plot = ['XGBoost Clinical', 'XGBoost Full', 'CNN 1D']
    test_f1 = [0.7429, 0.6512, 0.8966]
    test_auc = [0.9034, 0.9069, 0.9351]
    kfold_recall = [0.7368, 0.8553, 0.7883]

    fig = make_subplots(rows=1, cols=3, subplot_titles=['Test F1 Score', 'Test AUC-ROC', 'K-Fold Recall'])

    fig.add_trace(
        go.Bar(x=models_plot, y=test_f1, marker_color=['#e74c3c', '#2980b9', '#8e44ad'],
               text=[f'{v:.3f}' for v in test_f1], textposition='outside'),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=models_plot, y=test_auc, marker_color=['#e74c3c', '#2980b9', '#8e44ad'],
               text=[f'{v:.3f}' for v in test_auc], textposition='outside'),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=models_plot, y=kfold_recall, marker_color=['#e74c3c', '#2980b9', '#8e44ad'],
               text=[f'{v:.3f}' for v in kfold_recall], textposition='outside'),
        row=1, col=3,
    )

    fig.update_layout(
        height=400, showlegend=False,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_yaxes(range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    # Threshold analysis
    st.markdown("---")
    st.markdown("### 🎚️ Analisis Threshold Interaktif")
    st.markdown("Lihat bagaimana performa model berubah dengan threshold yang berbeda.")

    metadata = load_metadata()
    if metadata is not None and (cnn_data or xgb_data):
        n_sample = st.slider("Jumlah sampel untuk analisis", 10, min(100, len(metadata)), 30, key='thr_analysis')

        if st.button("📊 Analisis Threshold", key='run_thr'):
            sample_df = metadata.sample(n=n_sample, random_state=42)
            all_probs = {m: [] for m in ['CNN 1D', 'XGB Clinical', 'XGB Full'] if
                         (m.startswith('CNN') and cnn_data) or (m.startswith('XGB') and xgb_data)}
            all_true = []

            prog = st.progress(0)
            for i, (_, row) in enumerate(sample_df.iterrows()):
                try:
                    sig, _, fs_val = load_ecg_wfdb(row['patient_id'])
                    sig_f = bandpass_filter(sig, fs=fs_val)
                    sig_n = minmax_normalize(sig_f)
                    all_true.append(row['label'])

                    if cnn_data and 'CNN 1D' in all_probs:
                        pred = predict_cnn(sig_n, cnn_data)
                        all_probs['CNN 1D'].append(pred['probability'] if pred else 0.5)
                    if xgb_data and 'XGB Clinical' in all_probs:
                        pred = predict_xgboost(sig_f, xgb_data, 'm1')
                        all_probs['XGB Clinical'].append(pred['probability'] if pred else 0.5)
                    if xgb_data and 'XGB Full' in all_probs:
                        pred = predict_xgboost(sig_f, xgb_data, 'm2')
                        all_probs['XGB Full'].append(pred['probability'] if pred else 0.5)
                except Exception:
                    all_true.append(row['label'])
                    for k in all_probs:
                        all_probs[k].append(0.5)
                prog.progress((i + 1) / n_sample)
            prog.empty()

            all_true = np.array(all_true)
            from sklearn.metrics import f1_score as f1_fn, recall_score as rec_fn

            thresholds = np.arange(0.1, 0.9, 0.05)
            fig = make_subplots(rows=1, cols=2, subplot_titles=['F1 Score vs Threshold', 'Recall vs Threshold'])

            colors_iter = iter(['#e74c3c', '#2980b9', '#8e44ad'])
            for model_name, probs in all_probs.items():
                probs_arr = np.array(probs)
                f1s = [f1_fn(all_true, (probs_arr >= t).astype(int), zero_division=0) for t in thresholds]
                recs = [rec_fn(all_true, (probs_arr >= t).astype(int), zero_division=0) for t in thresholds]
                c = next(colors_iter)

                fig.add_trace(
                    go.Scatter(x=thresholds, y=f1s, name=model_name,
                               line=dict(color=c, width=2)),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=thresholds, y=recs, name=model_name,
                               line=dict(color=c, width=2), showlegend=False),
                    row=1, col=2,
                )

            fig.update_layout(
                height=400, paper_bgcolor='white', plot_bgcolor='white',
                margin=dict(l=40, r=20, t=40, b=40),
            )
            fig.update_xaxes(title='Threshold')
            fig.update_yaxes(title='Score', range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 15. PAGE: TENTANG
# ============================================================

def page_tentang():
    st.markdown(
        '<div class="main-header">'
        '<h1>ℹ️ Tentang BrugadaAI</h1>'
        '<p>Informasi tentang Sindrom Brugada dan sistem ini</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_brugada, tab_system, tab_guide = st.tabs([
        "🫀 Sindrom Brugada", "🤖 Tentang Sistem", "📖 Panduan Penggunaan"
    ])

    with tab_brugada:
        st.markdown("""
        ### Apa itu Sindrom Brugada?

        **Sindrom Brugada** adalah kelainan irama jantung (aritmia) yang jarang namun berpotensi
        mengancam jiwa. Sindrom ini ditandai oleh kelainan EKG yang khas dan peningkatan risiko
        kematian jantung mendadak (*sudden cardiac death*).

        #### Karakteristik EKG
        - **ST-segment elevation** tipe *coved* pada lead precordial kanan (V1–V3)
        - Sering disertai pola *right bundle branch block*
        - Dapat muncul spontan atau dipicu oleh *sodium channel blockers*

        #### Kriteria Diagnosis
        1. Pola EKG karakteristik (spontan atau induced)
        2. Riwayat sinkop (pingsan)
        3. Riwayat aritmia ventrikuler yang terdokumentasi
        4. Riwayat keluarga dengan kematian jantung mendadak

        #### Epidemiologi
        - Prevalensi: ~1-5 per 10.000 orang
        - Lebih sering pada pria (8-10× dibanding wanita)
        - Onset gejala usia rata-rata 40 tahun
        - Bertanggung jawab atas 4-12% kematian jantung mendadak

        #### Tatalaksana
        - **ICD (Implantable Cardioverter-Defibrillator)** untuk pasien risiko tinggi
        - Hindari obat-obatan yang dapat memperburuk (sodium channel blockers)
        - Monitoring rutin oleh kardiolog
        - Skrining genetik keluarga
        """)

    with tab_system:
        st.markdown("""
        ### Tentang BrugadaAI

        BrugadaAI adalah sistem skrining berbasis kecerdasan buatan yang dirancang untuk
        membantu deteksi dini Sindrom Brugada dari rekaman EKG 12-lead.

        #### Dataset
        - **Nama:** Brugada-HUCA v1.0.0
        - **Subjek:** 363 individu
        - **Sampling Rate:** 100 Hz
        - **Durasi:** 12 detik per rekaman
        - **Format:** WFDB (WaveForm DataBase)

        #### Model yang Digunakan

        | Model | Tipe | Input | Keunggulan |
        |-------|------|-------|------------|
        | CNN 1D | Deep Learning | Raw ECG Signal | Best overall performance |
        | CNN 1D + Aug | Deep Learning | Augmented ECG | More robust to noise |
        | XGBoost Clinical | Classical ML | 413 extracted features | Best F1 score |
        | XGBoost Full | Classical ML | 545 extracted features | Best Recall |

        #### Preprocessing
        1. **Bandpass Filter:** 0.5-40 Hz (Butterworth, order 4)
        2. **Normalisasi:** Min-Max per lead
        3. **Feature Extraction:** QRS, ST-segment, T-wave, PR, QT, RR interval

        #### Evaluasi
        - Stratified K-Fold Cross Validation (5 fold)
        - Metric: F1-Score, Recall, AUC-ROC
        - Threshold tuning per model

        ---

        ⚠️ **DISCLAIMER:** Sistem ini adalah alat bantu skrining dan BUKAN pengganti diagnosis
        dari dokter spesialis. Hasil harus selalu dikonfirmasi oleh kardiolog yang berkualifikasi.
        """)

    with tab_guide:
        st.markdown("""
        ### Panduan Penggunaan

        #### 1. Persiapan Model
        Sebelum menggunakan aplikasi, pastikan model sudah diekspor dari notebook:

        ```python
        # Jalankan di akhir notebook BRUGADA_CNN-collab.ipynb:
        import os, torch
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': best_model,
            'model_aug_state_dict': best_model_aug,
            'threshold': float(best_thr),
            'threshold_aug': float(best_thr_aug),
        }, 'models/cnn_models.pth')
        ```

        ```python
        # Jalankan di akhir notebook BRUGADA-Classical-Models.ipynb:
        import os, joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'xgb_m1': xgb_m1,
            'xgb_m2': xgb_m2,
            'scaler_c': scaler_c,
            'scaler_f': scaler_f,
            'threshold_m1': float(best_thr_m1),
            'threshold_m2': float(best_thr_m2),
        }, 'models/xgboost_models.pkl')
        ```

        #### 2. Menjalankan Aplikasi
        ```bash
        streamlit run app.py
        ```

        #### 3. Analisis ECG
        - **Upload:** Upload file .hea dan .dat dari rekaman WFDB
        - **Dataset:** Pilih pasien dari dataset yang sudah ada
        - Sistem akan otomatis memproses sinyal dan menjalankan semua model

        #### 4. Interpretasi Hasil
        - **Hijau (Normal):** Tidak terdeteksi pola Brugada
        - **Merah (Brugada):** Terdeteksi pola yang konsisten dengan Sindrom Brugada
        - **Ensemble:** Kombinasi prediksi dari semua model
        - **Risk Gauge:** Visualisasi probabilitas dengan threshold

        #### 5. Fitur Interaktif
        - 🔄 Ganti tampilan sinyal (Raw/Filtered/Normalized)
        - 📍 Pilih lead spesifik untuk detail
        - ⚙️ Atur threshold klasifikasi
        - 📄 Download laporan klinis
        - 📊 Batch prediction untuk multiple pasien
        """)

# ============================================================
# 16. MAIN APP ROUTER
# ============================================================

if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🔬 Analisis ECG":
    page_analisis_ecg()
elif page == "📂 Dataset Explorer":
    page_dataset_explorer()
elif page == "📊 Perbandingan Model":
    page_model_comparison()
elif page == "ℹ️ Tentang":
    page_tentang()
