"""
Export Models — Brugada Syndrome
================================
Jalankan kode di bawah ini di notebook masing-masing untuk menyimpan model terlatih.
Model akan disimpan ke folder 'models/' dan digunakan oleh app.py (Streamlit).

LANGKAH:
  1. Buka & jalankan semua cell di BRUGADA_CNN-collab.ipynb (pastikan training selesai)
  2. Copy-paste Block A di cell terakhir notebook CNN, lalu jalankan
  3. Buka & jalankan semua cell di BRUGADA-Classical-Models.ipynb (pastikan training selesai)
  4. Copy-paste Block B di cell terakhir notebook Classical, lalu jalankan
  5. Pastikan folder 'models/' berisi: cnn_models.pth dan xgboost_models.pkl
  6. Jalankan:  streamlit run app.py
"""

# ================================================================
# BLOCK A — Simpan di cell terakhir BRUGADA_CNN-collab.ipynb
# ================================================================

BLOCK_A = """
# === EXPORT CNN MODELS ===
import os, torch

os.makedirs('models', exist_ok=True)

torch.save({
    'model_state_dict': best_model,
    'model_aug_state_dict': best_model_aug,
    'threshold': float(best_thr),
    'threshold_aug': float(best_thr_aug),
}, 'models/cnn_models.pth')

print('✅ CNN models saved to models/cnn_models.pth')
print(f'   Threshold CNN       : {best_thr}')
print(f'   Threshold CNN + Aug : {best_thr_aug}')
"""

# ================================================================
# BLOCK B — Simpan di cell terakhir BRUGADA-Classical-Models.ipynb
# ================================================================

BLOCK_B = """
# === EXPORT XGBOOST MODELS ===
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

print('✅ XGBoost models saved to models/xgboost_models.pkl')
print(f'   Threshold Model 1 (Clinical) : {best_thr_m1}')
print(f'   Threshold Model 2 (Full)     : {best_thr_m2}')
"""

if __name__ == '__main__':
    print("=" * 60)
    print("  EXPORT MODELS — Brugada Syndrome")
    print("=" * 60)
    print()
    print("Copy-paste kode berikut ke cell terakhir notebook Anda:")
    print()
    print("─" * 60)
    print("BLOCK A — untuk BRUGADA_CNN-collab.ipynb:")
    print("─" * 60)
    print(BLOCK_A)
    print("─" * 60)
    print("BLOCK B — untuk BRUGADA-Classical-Models.ipynb:")
    print("─" * 60)
    print(BLOCK_B)
    print("─" * 60)
    print()
    print("Setelah kedua block dijalankan, start aplikasi dengan:")
    print("  streamlit run app.py")
    print()
