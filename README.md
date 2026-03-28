<p align="center">
  <h1 align="center">🫀 BrugadaAI — Brugada Syndrome ECG Classification</h1>
  <p align="center">
    AI-Based Brugada Syndrome Classification System<br>
    <em>AI-powered Brugada Syndrome Screening from 12-Lead ECG</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/XGBoost-2.x-006600?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/IIDSC-2026-orange" alt="IIDSC 2026">
</p>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [About Brugada Syndrome](#-about-brugada-syndrome)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results & Performance](#-results--performance)
- [File Structure](#-file-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Streamlit App](#-running-the-streamlit-app)
- [Streamlit Usage Guide](#-streamlit-usage-guide)
- [Technologies Used](#-technologies-used)
- [Disclaimer](#️-disclaimer)

---

## 🧬 About the Project

**BrugadaAI** is an automated screening system for detecting **Brugada Syndrome** from 12-lead ECG (Electrocardiography) recordings. This project combines **Deep Learning (1D CNN)** and **Classical Machine Learning (XGBoost)** approaches to provide robust and interpretable predictions.

The system includes an **interactive Streamlit-based web application** designed with a medical interface, allowing users to:
- Upload and analyze ECG recordings in WFDB format
- Obtain predictions from 4 models simultaneously (multi-model ensemble)
- Interactively visualize 12-lead ECG signals
- Download clinical screening reports

> **Context:** This project was built for the **IIDSC 2026** (Indonesia International Data Science Competition).

---

## 🫀 About Brugada Syndrome

**Brugada Syndrome** is a rare but potentially life-threatening genetic cardiac arrhythmia disorder. It is characterized by:

- **ST-segment elevation** with a *coved* pattern in the right precordial leads (**V1–V3**)
- Often accompanied by a *right bundle branch block* (RBBB) pattern
- Increased risk of **sudden cardiac death** (SCD)

### Key Facts
| Parameter | Value |
|-----------|-------|
| Prevalence | ~1–5 per 10,000 people |
| Male:Female Ratio | 8–10 : 1 |
| Average symptom onset | ~40 years |
| Contribution to SCD | 4–12% of sudden cardiac deaths |

### Diagnostic Criteria
1. Characteristic ECG pattern (spontaneous or drug-induced)
2. History of syncope (fainting)
3. Documented ventricular arrhythmia
4. Family history of sudden cardiac death

---

## 📊 Dataset

**Dataset:** Brugada-HUCA v1.0.0

| Parameter | Value |
|-----------|-------|
| Total Subjects | 363 individuals |
| Sampling Rate | 100 Hz |
| Recording Duration | 12 seconds (1200 samples) |
| Number of Leads | 12 standard ECG leads |
| Format | WFDB (.dat + .hea) |

### Class Distribution
| Class | Count | Percentage |
|-------|--------|------------|
| Normal (brugada=0) | 287 | 79.1% |
| Brugada (brugada>0) | 76 | 20.9% |

### Clinical Variables (metadata.csv)
| Variable | Description |
|----------|-------------|
| `patient_id` | Unique patient ID |
| `basal_pattern` | Pathological baseline ECG pattern (0/1) |
| `sudden_death` | Sudden death outcome variable (0/1) |
| `brugada` | Diagnosis label: 0=Normal, 1=Brugada, 2=Atypical |

---

## 🧠 Model Architecture

### Model 1 & 2 — XGBoost (Classical ML)

Uses manual **feature engineering** to extract clinical features from ECG signals.

| | Model 1 (Clinical) | Model 2 (Full) |
|---|---|---|
| **Number of Features** | 413 | 545 |
| **Features** | Statistics + QRS + ST + T-wave + RR | Statistics + QRS + ST + T-wave + **PR + QT** + RR |
| **Target** | Best F1 Score | Best Recall |
| **Scaler** | MinMaxScaler | MinMaxScaler |
| **Tuning** | GridSearchCV (5-Fold) | GridSearchCV (5-Fold) |

**Feature details per lead (12 leads):**
- **Statistics (11):** mean, std, min, max, range, skew, kurtosis, RMS, energy, dominant frequency, total power
- **QRS (9):** duration, amplitude, R/S ratio (mean, std, max, min)
- **ST-segment (9):** elevation, slope, area (mean, std, max, min)
- **T-wave (5):** amplitude, area, inversion (mean, std, count)
- **PR interval (5):** interval stats, P-wave amplitude *(Model 2 only)*
- **QT interval (6):** QT and QTc stats *(Model 2 only)*
- **RR interval (5):** mean, std, min, max, heart rate

### Model 3 & 4 — CNN 1D (Deep Learning)

Uses **raw** ECG signals (after filtering & normalization) without manual feature engineering.

```
Input: (batch, 12, 1200) — 12 leads × 1200 samples

Block 1: Conv1d(12→64, k=7) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
Block 2: Conv1d(64→128, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
Block 3: Conv1d(128→256, k=3) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
Block 4: Conv1d(256→256, k=3) → BatchNorm → ReLU → AdaptiveAvgPool1d(1)

Classifier: Flatten → Linear(256,128) → ReLU → Dropout(0.5) → Linear(128,1) → Sigmoid
```

| | CNN 1D | CNN 1D + Augmentation |
|---|---|---|
| **Input** | Raw ECG Signal (filtered + normalized) | Raw ECG + Minority class augmentation |
| **Parameters** | ~500,000 | ~500,000 |
| **Augmentation** | — | Gaussian noise, amplitude scaling, time shift, baseline wander |
| **Loss** | BCEWithLogitsLoss (weighted) | BCEWithLogitsLoss (weighted) |
| **Optimizer** | Adam (lr=1e-3, wd=1e-4) | Adam (lr=1e-3, wd=1e-4) |
| **Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau |

### Preprocessing Pipeline
1. **Bandpass Filter:** 0.5–40 Hz (Butterworth, order 4)
2. **Min-Max Normalization** per lead
3. **Transpose** to (channels, length) format for CNN

---

## 📈 Results & Performance

### Test Set (80/20 split, stratified)

| Model | Recall | F1 | AUC-ROC |
|-------|--------|----|---------|
| XGBoost Clinical | 0.8667 | 0.7429 | 0.9034 |
| XGBoost Full | 0.9333 | 0.6512 | 0.9069 |
| CNN 1D | 0.80 | 0.8966 | 0.9351 |
| **CNN 1D + Augmentation** | **0.93** | **0.9655** | **0.9420** |

### K-Fold Cross Validation (5 fold)

| Model | F1 (mean±std) | Recall (mean±std) | AUC (mean±std) |
|-------|---------------|-------------------|----------------|
| XGBoost Clinical | 0.6627 | 0.7368 | 0.8806 |
| XGBoost Full | 0.6047 | 0.8553 | 0.8802 |
| CNN 1D | 0.7182 | 0.7883 | 0.8635 |

> **Best overall:** CNN 1D (highest Test F1 & AUC-ROC)
> **Best recall:** XGBoost Full (highest sensitivity for screening)

---

## 📁 File Structure

```
BrugadaAI/
├── app.py                          # 🌐 Streamlit web application
├── export_models.py                # 📦 Script to export models from notebooks
├── requirements.txt                # 📋 Dependencies
├── README.md                       # 📖 Documentation (this file)
│
├── BRUGADA_CNN-collab.ipynb        # 🧠 Notebook: CNN 1D model (training & evaluation)
├── BRUGADA-Classical-Models.ipynb  # 🌲 Notebook: XGBoost model (training & evaluation)
├── Benchmark_Model_Notebook.ipynb  # 📊 Notebook: Benchmark reference
│
├── models/                         # 💾 Trained models (exported)
│   ├── cnn_models.pth              #     CNN 1D & CNN 1D+Aug (PyTorch)
│   └── xgboost_models.pkl          #     XGBoost Clinical & Full (joblib)
│
├── metadata.csv                    # 📋 Patient clinical data
├── metadata_dictionary.csv         # 📖 Metadata variable dictionary
├── RECORDS                         # 📝 Patient ID list
│
├── files/                          # 📂 Per-patient ECG data (WFDB format)
│   ├── 188981/
│   │   ├── 188981.dat              #     ECG signal (binary)
│   │   └── 188981.hea              #     Recording header metadata
│   ├── 251972/
│   │   ├── 251972.dat
│   │   └── 251972.hea
│   └── ... (363 patients)
│
├── cnn1d_results.png               # 📊 CNN visualization results
├── cnn1d_aug_results.png           # 📊 CNN + Augmentation visualization results
├── model1_clinical_results.png     # 📊 XGBoost Clinical visualization results
├── model2_full_results.png         # 📊 XGBoost Full visualization results
├── model_comparison_roc.png        # 📊 Model comparison ROC curve
│
├── LICENSE.txt                     # ⚖️ License
└── SHA256SUMS.txt                  # 🔒 File checksums
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/<username>/BrugadaAI.git
cd BrugadaAI
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
| Package | Version | Purpose |
|---------|-------|--------|
| streamlit | ≥1.30.0 | Web application framework |
| torch | ≥2.0.0 | CNN 1D deep learning |
| xgboost | ≥2.0.0 | Gradient boosting classifier |
| wfdb | ≥4.1.0 | Read WFDB ECG format |
| plotly | ≥5.18.0 | Interactive visualization |
| scipy | ≥1.11.0 | Signal processing |
| scikit-learn | ≥1.3.0 | Preprocessing & metrics |
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| joblib | ≥1.3.0 | Model serialization |

### 4. Ensure Models Are Available

Trained models must be in the `models/` folder:
```
models/
├── cnn_models.pth        # CNN models (PyTorch)
└── xgboost_models.pkl    # XGBoost models (joblib)
```

> If models are not available, you need to run the training notebooks first (see [Re-training Models](#re-training-models-optional)).

---

## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

To stop the application, press `Ctrl+C` in the terminal.

---

## 📖 Streamlit Usage Guide

The BrugadaAI application has **5 main pages** accessible through the sidebar navigation:

### 🏠 1. Dashboard

The main page displays:
- **Dataset summary** — total patients, normal vs. Brugada count
- **Model status** — which models were successfully loaded
- **Pie chart** of class distribution
- **System information** and ECG parameters

### 🔬 2. ECG Analysis (Main Page)

This is the core page of the application. How to use it:

#### Step 1 — Select ECG Data Source

There are 2 ways to input data:

| Tab | Method | Description |
|-----|--------|-------------|
| **📤 Upload WFDB File** | Upload `.hea` and `.dat` files | For new ECG data from outside the dataset |
| **📂 Select from Dataset** | Choose patient ID from dropdown | To test with existing data |

> **Note:** When uploading, both files (.hea and .dat) must have the **same name** (e.g., `188981.hea` & `188981.dat`).

#### Step 2 — 12-Lead ECG Visualization

After data is loaded, the system displays:
- **Interactive 12-lead ECG** — zoom, pan, hover to see details
- **3 display modes:** Raw (original), Filtered (after bandpass), Normalized (min-max)
- **Per-lead detail** — select specific leads (default: V1, V2, V3 as most relevant for Brugada)
- **R-Peak detection** — toggle to view R-peak locations

#### Step 3 — Automatic ECG Parameters

The system automatically calculates:
- Heart Rate (bpm)
- RR Interval (mean, std, min, max) in milliseconds

#### Step 4 — Classification Results

The system runs **all available models** and displays:
- **Result card per model** — green (Normal) or red (Brugada)
- **Probability** and **threshold** per model
- **Ensemble result** — consensus from all models:
  - ✅ NORMAL — no indication
  - ⚠️ CAUTION — some models detected Brugada
  - 🚨 POSITIVE — all models detected Brugada
- **Ground truth** shown if data is from the dataset (for validation)

#### Step 5 — Prediction Visualization

2 visualization tabs:
- **🎯 Risk Gauge** — risk meter per model (green→yellow→red)
- **📊 Comparison** — bar chart comparing probabilities across models

#### Step 6 — Threshold Settings (Optional)

Open the **⚙️ Threshold Settings** panel to:
- Adjust classification threshold per model with slider (0.0–1.0)
- Lower threshold = more sensitive (more positive detections)
- Higher threshold = more specific (fewer false positives)

#### Step 7 — Download Clinical Report

- **Preview** the report before downloading
- **Download** as a `.txt` file containing all classification results and medical disclaimer

### 📂 3. Dataset Explorer

Dataset exploration features:
- **Filter** by: class (Normal/Brugada), basal pattern, sudden death
- **Complete data table** with per-patient status
- **Distribution charts** for class and Brugada grade
- **Quick ECG Preview** — select a patient and instantly view their ECG
- **Batch Prediction** — run predictions on 5–50 patients at once, with progress bar and CSV result download

### 📊 4. Model Comparison

Reference page displaying:
- **Complete specifications** for each model (type, input, features)
- **CNN architecture** (layer-by-layer diagram)
- **XGBoost feature pipeline** (details for each feature type)
- **Performance table** from training results
- **Bar chart** comparing F1, AUC-ROC, Recall
- **Interactive Threshold Analysis** — run predictions on sample data and view F1/Recall vs Threshold curves for each model

### ℹ️ 5. About

Comprehensive information about:
- What is Brugada Syndrome (medical)
- Technical details of the BrugadaAI system
- Step-by-step usage guide

---

## Re-training Models (Optional)

If you want to retrain the models from scratch:

### 1. Run the CNN Notebook
Open `BRUGADA_CNN-collab.ipynb` and run all cells from top to bottom. The last cell will export the model to `models/cnn_models.pth`.

### 2. Run the XGBoost Notebook
Open `BRUGADA-Classical-Models.ipynb` and run all cells. The last cell will export the model to `models/xgboost_models.pkl`.

### 3. Verify
```bash
ls models/
# Output: cnn_models.pth  xgboost_models.pkl
```

---

## 🛠 Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.x |
| **Classical ML** | XGBoost 2.x, scikit-learn |
| **Signal Processing** | SciPy (Butterworth filter, Welch PSD, peak detection) |
| **ECG Data** | WFDB (WaveForm DataBase format) |
| **Web App** | Streamlit |
| **Visualization** | Plotly (interactive), Matplotlib (notebook) |
| **Data** | Pandas, NumPy |

---

## ⚕️ Disclaimer

> **IMPORTANT:** The BrugadaAI system is an AI-based **screening tool** and is **NOT** a substitute for professional medical diagnosis. Prediction results should always be confirmed by a qualified **cardiologist**. Do not make medical decisions based solely on the output of this system.

---

<p align="center">
  <b>BrugadaAI</b> — IIDSC 2026<br>
  <em>AI-powered Brugada Syndrome Screening System</em>
</p>
