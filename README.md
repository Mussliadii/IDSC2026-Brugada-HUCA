<p align="center">
  <h1 align="center">🫀 BrugadaAI — Brugada Syndrome ECG Classification</h1>
  <p align="center">
    Sistem Klasifikasi Sindrom Brugada Berbasis Kecerdasan Buatan<br>
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

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Tentang Sindrom Brugada](#-tentang-sindrom-brugada)
- [Dataset](#-dataset)
- [Arsitektur Model](#-arsitektur-model)
- [Hasil & Performa](#-hasil--performa)
- [Struktur File](#-struktur-file)
- [Instalasi & Setup](#-instalasi--setup)
- [Cara Menjalankan Streamlit](#-cara-menjalankan-streamlit)
- [Panduan Penggunaan Streamlit](#-panduan-penggunaan-streamlit)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Disclaimer](#️-disclaimer)

---

## 🧬 Tentang Proyek

**BrugadaAI** adalah sistem skrining otomatis untuk mendeteksi **Sindrom Brugada** dari rekaman EKG (Elektrokardiografi) 12-lead. Proyek ini menggabungkan pendekatan **Deep Learning (CNN 1D)** dan **Classical Machine Learning (XGBoost)** untuk memberikan prediksi yang robust dan dapat diinterpretasi.

Sistem ini dilengkapi dengan **aplikasi web interaktif berbasis Streamlit** yang dirancang dengan tampilan kedokteran, memungkinkan pengguna untuk:
- Upload dan menganalisis rekaman EKG dalam format WFDB
- Mendapatkan prediksi dari 4 model sekaligus (ensemble multi-model)
- Memvisualisasikan sinyal EKG 12-lead secara interaktif
- Mengunduh laporan klinis hasil skrining

> **Konteks:** Proyek ini dibuat untuk kompetisi **IIDSC 2026** (Indonesia International Data Science Competition).

---

## 🫀 Tentang Sindrom Brugada

**Sindrom Brugada** adalah kelainan irama jantung (aritmia) genetik yang jarang namun berpotensi mengancam jiwa. Sindrom ini ditandai oleh:

- **ST-segment elevation** tipe *coved* pada lead precordial kanan (**V1–V3**)
- Sering disertai pola *right bundle branch block* (RBBB)
- Peningkatan risiko **kematian jantung mendadak** (*sudden cardiac death*)

### Fakta Penting
| Parameter | Nilai |
|-----------|-------|
| Prevalensi | ~1–5 per 10.000 orang |
| Rasio Pria:Wanita | 8–10 : 1 |
| Onset gejala rata-rata | ~40 tahun |
| Kontribusi ke SCD | 4–12% kematian jantung mendadak |

### Kriteria Diagnosis
1. Pola EKG karakteristik (spontan atau drug-induced)
2. Riwayat sinkop (pingsan)
3. Aritmia ventrikuler yang terdokumentasi
4. Riwayat keluarga dengan kematian jantung mendadak

---

## 📊 Dataset

**Dataset:** Brugada-HUCA v1.0.0

| Parameter | Nilai |
|-----------|-------|
| Total Subjek | 363 individu |
| Sampling Rate | 100 Hz |
| Durasi Rekaman | 12 detik (1200 sampel) |
| Jumlah Lead | 12 lead EKG standar |
| Format | WFDB (.dat + .hea) |

### Distribusi Kelas
| Kelas | Jumlah | Persentase |
|-------|--------|------------|
| Normal (brugada=0) | 287 | 79.1% |
| Brugada (brugada>0) | 76 | 20.9% |

### Variabel Klinis (metadata.csv)
| Variabel | Deskripsi |
|----------|-----------|
| `patient_id` | ID unik pasien |
| `basal_pattern` | Pola EKG baseline patologis (0/1) |
| `sudden_death` | Variabel outcome kematian mendadak (0/1) |
| `brugada` | Label diagnosis: 0=Normal, 1=Brugada, 2=Atipikal |

---

## 🧠 Arsitektur Model

### Model 1 & 2 — XGBoost (Classical ML)

Menggunakan **feature engineering** manual untuk mengekstrak fitur klinis dari sinyal EKG.

| | Model 1 (Clinical) | Model 2 (Full) |
|---|---|---|
| **Jumlah Fitur** | 413 | 545 |
| **Fitur** | Statistik + QRS + ST + T-wave + RR | Statistik + QRS + ST + T-wave + **PR + QT** + RR |
| **Target** | Best F1 Score | Best Recall |
| **Scaler** | MinMaxScaler | MinMaxScaler |
| **Tuning** | GridSearchCV (5-Fold) | GridSearchCV (5-Fold) |

**Detail fitur per lead (12 lead):**
- **Statistik (11):** mean, std, min, max, range, skew, kurtosis, RMS, energy, dominant frequency, total power
- **QRS (9):** duration, amplitude, R/S ratio (mean, std, max, min)
- **ST-segment (9):** elevation, slope, area (mean, std, max, min)
- **T-wave (5):** amplitude, area, inversion (mean, std, count)
- **PR interval (5):** interval stats, P-wave amplitude *(Model 2 only)*
- **QT interval (6):** QT and QTc stats *(Model 2 only)*
- **RR interval (5):** mean, std, min, max, heart rate

### Model 3 & 4 — CNN 1D (Deep Learning)

Menggunakan sinyal EKG **mentah** (setelah filtering & normalisasi) tanpa feature engineering manual.

```
Input: (batch, 12, 1200) — 12 leads × 1200 samples

Block 1: Conv1d(12→64, k=7) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
Block 2: Conv1d(64→128, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
Block 3: Conv1d(128→256, k=3) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
Block 4: Conv1d(256→256, k=3) → BatchNorm → ReLU → AdaptiveAvgPool1d(1)

Classifier: Flatten → Linear(256,128) → ReLU → Dropout(0.5) → Linear(128,1) → Sigmoid
```

| | CNN 1D | CNN 1D + Augmentasi |
|---|---|---|
| **Input** | Raw ECG Signal (filtered + normalized) | Raw ECG + Augmentasi kelas minoritas |
| **Parameter** | ~500.000 | ~500.000 |
| **Augmentasi** | — | Gaussian noise, amplitude scaling, time shift, baseline wander |
| **Loss** | BCEWithLogitsLoss (weighted) | BCEWithLogitsLoss (weighted) |
| **Optimizer** | Adam (lr=1e-3, wd=1e-4) | Adam (lr=1e-3, wd=1e-4) |
| **Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau |

### Preprocessing Pipeline
1. **Bandpass Filter:** 0.5–40 Hz (Butterworth, order 4)
2. **Min-Max Normalization** per lead
3. **Transpose** ke format (channels, length) untuk CNN

---

## 📈 Hasil & Performa

### Test Set (80/20 split, stratified)

| Model | Recall | F1 | AUC-ROC |
|-------|--------|----|---------|
| XGBoost Clinical | 0.8667 | 0.7429 | 0.9034 |
| XGBoost Full | 0.9333 | 0.6512 | 0.9069 |
| CNN 1D | 0.80 | 0.8966 | 0.9351 |
| **CNN 1D + Augmentasi** | **0.93** | **0.9655** | **0.9420** |

### K-Fold Cross Validation (5 fold)

| Model | F1 (mean±std) | Recall (mean±std) | AUC (mean±std) |
|-------|---------------|-------------------|----------------|
| XGBoost Clinical | 0.6627 | 0.7368 | 0.8806 |
| XGBoost Full | 0.6047 | 0.8553 | 0.8802 |
| CNN 1D | 0.7182 | 0.7883 | 0.8635 |

> **Best overall:** CNN 1D (highest Test F1 & AUC-ROC)
> **Best recall:** XGBoost Full (highest sensitivity for screening)

---

## 📁 Struktur File

```
BrugadaAI/
├── app.py                          # 🌐 Streamlit web application
├── export_models.py                # 📦 Script untuk export model dari notebook
├── requirements.txt                # 📋 Dependencies
├── README.md                       # 📖 Dokumentasi (file ini)
│
├── BRUGADA_CNN-collab.ipynb        # 🧠 Notebook: CNN 1D model (training & evaluation)
├── BRUGADA-Classical-Models.ipynb  # 🌲 Notebook: XGBoost model (training & evaluation)
├── Benchmark_Model_Notebook.ipynb  # 📊 Notebook: Benchmark reference
│
├── models/                         # 💾 Model terlatih (hasil export)
│   ├── cnn_models.pth              #     CNN 1D & CNN 1D+Aug (PyTorch)
│   └── xgboost_models.pkl          #     XGBoost Clinical & Full (joblib)
│
├── metadata.csv                    # 📋 Data klinis pasien
├── metadata_dictionary.csv         # 📖 Kamus variabel metadata
├── RECORDS                         # 📝 Daftar patient ID
│
├── files/                          # 📂 Data EKG per pasien (WFDB format)
│   ├── 188981/
│   │   ├── 188981.dat              #     Sinyal EKG (binary)
│   │   └── 188981.hea              #     Header metadata rekaman
│   ├── 251972/
│   │   ├── 251972.dat
│   │   └── 251972.hea
│   └── ... (363 pasien)
│
├── cnn1d_results.png               # 📊 Hasil visualisasi CNN
├── cnn1d_aug_results.png           # 📊 Hasil visualisasi CNN + Augmentasi
├── model1_clinical_results.png     # 📊 Hasil visualisasi XGBoost Clinical
├── model2_full_results.png         # 📊 Hasil visualisasi XGBoost Full
├── model_comparison_roc.png        # 📊 ROC curve perbandingan model
│
├── LICENSE.txt                     # ⚖️ Lisensi
└── SHA256SUMS.txt                  # 🔒 Checksum file
```

---

## ⚙️ Instalasi & Setup

### 1. Clone Repository

```bash
git clone https://github.com/<username>/BrugadaAI.git
cd BrugadaAI
```

### 2. Buat Virtual Environment (opsional tapi direkomendasikan)

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

**Dependencies utama:**
| Package | Versi | Fungsi |
|---------|-------|--------|
| streamlit | ≥1.30.0 | Web application framework |
| torch | ≥2.0.0 | CNN 1D deep learning |
| xgboost | ≥2.0.0 | Gradient boosting classifier |
| wfdb | ≥4.1.0 | Baca format EKG WFDB |
| plotly | ≥5.18.0 | Visualisasi interaktif |
| scipy | ≥1.11.0 | Signal processing |
| scikit-learn | ≥1.3.0 | Preprocessing & metrics |
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| joblib | ≥1.3.0 | Model serialization |

### 4. Pastikan Model Sudah Ada

Model terlatih harus berada di folder `models/`:
```
models/
├── cnn_models.pth        # CNN models (PyTorch)
└── xgboost_models.pkl    # XGBoost models (joblib)
```

> Jika model belum ada, Anda perlu menjalankan notebook training terlebih dahulu (lihat bagian [Re-training Model](#re-training-model-opsional)).

---

## 🚀 Cara Menjalankan Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

Untuk menghentikan aplikasi, tekan `Ctrl+C` di terminal.

---

## 📖 Panduan Penggunaan Streamlit

Aplikasi BrugadaAI memiliki **5 halaman utama** yang dapat diakses melalui sidebar navigasi:

### 🏠 1. Dashboard

Halaman utama menampilkan:
- **Ringkasan dataset** — total pasien, jumlah normal vs. Brugada
- **Status model** — model mana saja yang berhasil dimuat
- **Pie chart** distribusi kelas
- **Informasi sistem** dan parameter EKG

### 🔬 2. Analisis ECG (Halaman Utama)

Halaman ini adalah inti dari aplikasi. Cara menggunakannya:

#### Langkah 1 — Pilih Sumber Data EKG

Ada 2 cara memasukkan data:

| Tab | Cara | Keterangan |
|-----|------|------------|
| **📤 Upload File WFDB** | Upload file `.hea` dan `.dat` | Untuk data EKG baru dari luar dataset |
| **📂 Pilih dari Dataset** | Pilih patient ID dari dropdown | Untuk menguji dengan data yang sudah ada |

> **Catatan:** Saat upload, kedua file (.hea dan .dat) harus memiliki **nama yang sama** (contoh: `188981.hea` & `188981.dat`).

#### Langkah 2 — Visualisasi EKG 12-Lead

Setelah data dimuat, sistem menampilkan:
- **EKG 12-lead interaktif** — zoom, pan, hover untuk lihat detail
- **3 mode tampilan:** Raw (asli), Filtered (setelah bandpass), Normalized (min-max)
- **Detail per lead** — pilih lead spesifik (default: V1, V2, V3 karena paling relevan untuk Brugada)
- **Deteksi R-Peak** — toggle untuk melihat lokasi puncak R

#### Langkah 3 — Parameter EKG Otomatis

Sistem menghitung otomatis:
- Heart Rate (bpm)
- RR Interval (mean, std, min, max) dalam milidetik

#### Langkah 4 — Hasil Klasifikasi

Sistem menjalankan **semua model yang tersedia** dan menampilkan:
- **Kartu hasil per model** — warna hijau (Normal) atau merah (Brugada)
- **Probabilitas** dan **threshold** per model
- **Ensemble result** — konsensus dari semua model:
  - ✅ NORMAL — tidak ada indikasi
  - ⚠️ WASPADA — sebagian model mendeteksi Brugada
  - 🚨 POSITIF — semua model mendeteksi Brugada
- **Ground truth** ditampilkan jika data dari dataset (untuk validasi)

#### Langkah 5 — Visualisasi Prediksi

2 tab visualisasi:
- **🎯 Risk Gauge** — meter risiko per model (hijau→kuning→merah)
- **📊 Comparison** — bar chart perbandingan probabilitas antar model

#### Langkah 6 — Pengaturan Threshold (Opsional)

Buka panel **⚙️ Pengaturan Threshold** untuk:
- Menyesuaikan threshold klasifikasi per model dengan slider (0.0–1.0)
- Threshold lebih rendah = lebih sensitif (lebih banyak deteksi positif)
- Threshold lebih tinggi = lebih spesifik (lebih sedikit false positive)

#### Langkah 7 — Download Laporan Klinis

- **Preview** laporan sebelum download
- **Download** sebagai file `.txt` yang berisi semua hasil klasifikasi dan disclaimer medis

### 📂 3. Dataset Explorer

Fitur eksplorasi dataset:
- **Filter** berdasarkan: kelas (Normal/Brugada), basal pattern, sudden death
- **Tabel data** lengkap dengan status per pasien
- **Grafik distribusi** kelas dan grade Brugada
- **Quick ECG Preview** — pilih pasien dan langsung lihat EKG-nya
- **Batch Prediction** — jalankan prediksi pada 5–50 pasien sekaligus, dengan progress bar dan download CSV hasil

### 📊 4. Perbandingan Model

Halaman referensi yang menampilkan:
- **Spesifikasi** lengkap tiap model (tipe, input, fitur)
- **Arsitektur CNN** (diagram layer-by-layer)
- **Pipeline fitur XGBoost** (detail setiap jenis fitur)
- **Tabel performa** dari hasil training
- **Bar chart** perbandingan F1, AUC-ROC, Recall
- **Analisis Threshold Interaktif** — jalankan prediksi pada sampel data dan lihat kurva F1/Recall vs Threshold untuk setiap model

### ℹ️ 5. Tentang

Informasi lengkap tentang:
- Apa itu Sindrom Brugada (medis)
- Detail teknis sistem BrugadaAI
- Panduan penggunaan step-by-step

---

## Re-training Model (Opsional)

Jika ingin melatih ulang model dari awal:

### 1. Jalankan Notebook CNN
Buka `BRUGADA_CNN-collab.ipynb` dan jalankan semua cell dari atas ke bawah. Cell terakhir akan meng-export model ke `models/cnn_models.pth`.

### 2. Jalankan Notebook XGBoost
Buka `BRUGADA-Classical-Models.ipynb` dan jalankan semua cell. Cell terakhir akan meng-export model ke `models/xgboost_models.pkl`.

### 3. Verifikasi
```bash
ls models/
# Output: cnn_models.pth  xgboost_models.pkl
```

---

## 🛠 Teknologi yang Digunakan

| Kategori | Teknologi |
|----------|-----------|
| **Bahasa** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.x |
| **Classical ML** | XGBoost 2.x, scikit-learn |
| **Signal Processing** | SciPy (Butterworth filter, Welch PSD, peak detection) |
| **ECG Data** | WFDB (WaveForm DataBase format) |
| **Web App** | Streamlit |
| **Visualisasi** | Plotly (interaktif), Matplotlib (notebook) |
| **Data** | Pandas, NumPy |

---

## ⚕️ Disclaimer

> **PENTING:** Sistem BrugadaAI adalah **alat bantu skrining** berbasis kecerdasan buatan dan **BUKAN** pengganti diagnosis medis profesional. Hasil prediksi harus selalu dikonfirmasi oleh **dokter spesialis jantung (kardiolog)** yang berkualifikasi. Jangan mengambil keputusan medis hanya berdasarkan output dari sistem ini.

---

<p align="center">
  <b>BrugadaAI</b> — IIDSC 2026<br>
  <em>AI-powered Brugada Syndrome Screening System</em>
</p>
