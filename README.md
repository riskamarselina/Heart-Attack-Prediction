# 🫀 Sistem Deteksi Risiko Penyakit Jantung

Aplikasi berbasis Machine Learning untuk membantu masyarakat awam mendeteksi risiko serangan jantung berdasarkan data kesehatan pribadi.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📌 Fitur Utama

- **Prediksi Risiko** — Ya/Tidak berisiko serangan jantung beserta tingkat probabilitasnya
- **Interpretasi Variabel** — Keterangan status setiap data yang dimasukkan (normal, waspada, bahaya)
- **Rekomendasi Makanan** — Saran makanan yang dianjurkan dan dihindari sesuai kondisi kesehatan

---

## 📊 Dataset

Dataset: **Heart Attack Prediction Indonesia**  
- 158.355 baris data pasien  
- 27 fitur (demografi, gaya hidup, klinis)  
- Target: `heart_attack` (0 = Tidak, 1 = Ya)  

---

## 🤖 Model Machine Learning

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | ~72% | ~0.68 | ~0.79 |
| Random Forest | ~82% | ~0.79 | ~0.90 |
| **XGBoost** ✅ | **~85%** | **~0.83** | **~0.92** |

Model terbaik: **XGBoost** — disimpan di `model_artifacts/`

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/heart-attack-system.git
cd heart-attack-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Siapkan Model Artifacts
Jalankan notebook `heart_attack_prediction.ipynb` di Google Colab terlebih dahulu, lalu download `model_artifacts.zip` dan ekstrak ke folder ini.

Struktur folder yang dibutuhkan:
```
heart-attack-system/
├── app.py
├── requirements.txt
├── heart_attack_prediction.ipynb
├── model_artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── metadata.json
└── README.md
```

### 4. Jalankan Streamlit
```bash
streamlit run app.py
```

---

## ☁️ Deploy ke Streamlit Cloud

1. Push semua file ke GitHub (termasuk `model_artifacts/`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → pilih `app.py` → Deploy

> **Catatan:** Folder `model_artifacts/` harus ikut di-push ke GitHub agar Streamlit Cloud bisa membaca modelnya.

---

## 🧪 Variabel Input

| Variabel | Tipe | Keterangan |
|---|---|---|
| Usia | Numerik | Usia dalam tahun |
| Jenis Kelamin | Kategori | Male / Female |
| Tekanan Darah | Numerik | Sistolik & Diastolik (mmHg) |
| Gula Darah Puasa | Numerik | mg/dL |
| Kolesterol HDL/LDL | Numerik | mg/dL |
| Trigliserida | Numerik | mg/dL |
| Lingkar Pinggang | Numerik | cm |
| Status Merokok | Kategori | Never / Past / Current |
| Aktivitas Fisik | Kategori | Low / Moderate / High |
| Tingkat Stres | Kategori | Low / Moderate / High |
| Hasil EKG | Kategori | Normal / Abnormal |
| ... | ... | ... |

---

## ⚕️ Disclaimer

> Aplikasi ini bersifat **informatif** dan **bukan pengganti diagnosis medis resmi**. Selalu konsultasikan kondisi kesehatan Anda dengan dokter atau tenaga medis yang berwenang.

---

## 📝 Lisensi

MIT License — bebas digunakan untuk keperluan edukasi.
