---
nama website: BISINDO Live
deskripsi: Klasifikasi huruf BISINDO real-time dari webcam
---

# BISINDO Landmark Classifier

Klasifikasi 26 huruf BISINDO (A–Z) dari landmark tangan MediaPipe dengan SVM, Random Forest, dan KNN, lalu di-deploy sebagai website real-time berarsitektur client–server.

Alur proyek:

1. **`BISINDO_Training.ipynb`** — preprocessing, ekstraksi fitur, training, evaluasi, menyimpan model terbaik ke `models/`
2. **`backend/`** — FastAPI memuat model tersebut dan menyajikan REST API `/predict`
3. **`frontend/`** — website webcam yang memanggil API secara real-time
4. **`Dockerfile`** — membungkus server untuk di-hosting di **Railway** (lihat `DEPLOY.md`); `jalankan_demo.bat` sebagai cadangan (laptop + Cloudflare Tunnel)

## Struktur

```
PROJECT BISINDO/
├── bisindo/images/{train,val}/A..Z/*.jpg   <- dataset
├── .venv/                                  <- virtual environment Python
└── bisindo_ml/
    ├── BISINDO_Training.ipynb   # Bagian 1-2: preprocessing, training, evaluasi, simpan model
    ├── requirements.txt         # library untuk notebook + backend
    ├── src/
    │   ├── config.py               # path & konstanta (dipakai backend)
    │   └── landmark_extractor.py   # MediaPipe + normalisasi (dipakai backend, logika = notebook)
    ├── backend/
    │   ├── main.py                 # Bagian 3: FastAPI (/health, /predict, /predict/upload)
    │   └── test_api.py             # uji API dengan gambar dari dataset
    ├── frontend/                   # Bagian 4: index.html, style.css, app.js, logo
    ├── assets/                     # logo PNG/SVG untuk laporan & slide
    ├── Dockerfile, requirements-deploy.txt   # container server untuk Railway
    ├── DEPLOY.md                # panduan deployment & uji deployment
    ├── jalankan_demo.bat        # cadangan: server di laptop + Cloudflare Tunnel
    ├── data/      (dibuat notebook)
    ├── models/    (dibuat notebook)
    └── reports/   (dibuat notebook)
```

## 1. Persiapan (dijalankan sekali saja)

Dari folder `PROJECT BISINDO`:

```powershell
python -m venv .venv
.venv\Scripts\activate
cd bisindo_ml
pip install -r requirements.txt
```

## 2. Training model lewat notebook

1. Buka `BISINDO_Training.ipynb` di VS Code (atau Jupyter).
2. Pilih kernel **`.venv`** di pojok kanan atas: **Select Kernel → Python Environments → `.venv`**. Pasang `ipykernel` bila diminta.
3. Atur sel konfigurasi bila perlu:
   - `REUSE_FEATURES = True` — pakai `data/landmarks.csv` yang sudah ada (lewati ekstraksi ±25 menit). Set `False` untuk mengekstrak ulang dari gambar.
   - `QUICK_GRID = True` — grid kecil untuk uji cepat.
   - `LIMIT_PER_CLASS = 5` — hanya 5 gambar per huruf untuk uji cepat ekstraksi.
4. **Run All.**

Isi notebook:

| Bagian | Isi |
|---|---|
| 1 | Eksplorasi dataset (jumlah per huruf, contoh gambar) |
| 2 | Deteksi landmark MediaPipe + median blur untuk gambar ber-noise |
| 3 | Normalisasi relatif wrist & vektor 126 fitur (zero-padding 1 tangan) |
| 4 | Ekstraksi seluruh dataset + detection rate per huruf |
| 5 | Label encoding, stratified split 80:20, StandardScaler dalam Pipeline |
| 6 | GridSearchCV 5-fold: SVM, Random Forest, KNN |
| 7 | Evaluasi: accuracy, precision, recall, macro-F1 |
| 8 | Confusion matrix & pasangan huruf yang sering tertukar |
| 9 | **Simpan model terbaik** ke `models/` |
| 10–11 | Uji inferensi end-to-end & ringkasan |

Model terbaik dipilih berdasarkan skor CV macro-F1. Pada run penuh, hasilnya adalah **SVM** (RBF, C=100): test accuracy 0,983 dan macro-F1 0,983.

Pada run pertama, `hand_landmarker.task` (~8 MB) diunduh otomatis ke `models/`, jadi butuh internet.

### Output notebook

| File | Isi |
|---|---|
| `models/best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `model_meta.json` | **Artefak yang dibaca backend** |
| `models/hand_landmarker.task` | Model MediaPipe Hand Landmarker |
| `models/all/<Model>_pipeline.joblib` | Pipeline lengkap ketiga model (tidak di-upload ke GitHub) |
| `data/landmarks.csv`, `data/failed_images.csv` | Fitur per gambar & gambar yang gagal terdeteksi |
| `reports/extraction_summary.csv` | Detection rate per huruf, jumlah 1 vs 2 tangan |
| `reports/model_comparison.csv/.png` | Perbandingan metrik ketiga model |
| `reports/confusion_matrices.png` | Confusion matrix ternormalisasi ketiga model |
| `reports/classification_report_<Model>.txt` | Precision/recall/F1 per huruf |
| `reports/top_confusions_<Model>.csv` | Pasangan huruf yang paling sering tertukar |
| `reports/cv_results_<Model>.csv` | Skor semua kombinasi hyperparameter |

> **Penting:** backend memakai `src/landmark_extractor.py` dengan logika yang identik dengan notebook (median blur, normalisasi relatif wrist, urutan tangan kiri→kanan, zero-padding). Jika logika fitur di notebook diubah, ubah juga file ini lalu latih ulang. Tanpa itu, fitur saat prediksi tidak cocok dengan fitur saat training.

## 3. Backend (Bagian 3)

Setelah notebook selesai dan `models/` terisi, jalankan dari folder `bisindo_ml`:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- Website: **http://localhost:8000**
- Swagger UI: http://localhost:8000/docs (bisa upload gambar lewat `/predict/upload`)
- Uji otomatis (terminal kedua): `python backend/test_api.py`

Contoh respons `POST /predict`:

```json
{"hand_detected": true, "num_hands": 2, "prediction": "A", "confidence": 0.97,
 "top_k": [{"label": "A", "confidence": 0.97}, ...], "landmarks": [[[x, y], ...21 titik], ...], "inference_ms": 45.2}
```

## 4. Frontend (Bagian 4)

Disajikan oleh FastAPI di `/app/`. Buka http://localhost:8000, klik **Mulai Kamera**, lalu izinkan akses kamera.

- Frame dikirim ±5 fps (dapat diatur), diperkecil ke lebar 320 px, satu request pada satu waktu.
- Anti-flicker: majority voting atas 7 prediksi terakhir; prediksi di bawah ambang confidence dihitung "ragu".
- Huruf yang stabil ±1 detik otomatis ditambahkan ke *Teks tersusun* dan *Riwayat*.
- Label "Tangan tidak terdeteksi" muncul bila tidak ada tangan di kamera.
- Video di-mirror lewat CSS saja. Frame yang dikirim ke server tidak di-mirror, supaya konsisten dengan data training.

## 5. Deployment (link publik)

**Utama: Railway** (Trial $5 / 30 hari, tanpa kartu kredit, link tetap `https://<nama>.up.railway.app`):

1. Upload folder `bisindo_ml` ke GitHub (repo `bisindo-live`), termasuk `models/`.
2. Di https://railway.com: **Login with GitHub** → **New Project** → **Deploy from GitHub repo** → pilih `bisindo-live`.
3. Tambah variabel `PORT` = `7860`, lalu **Settings → Networking → Generate Domain** (port `7860`).
4. Buka `https://<nama>.up.railway.app`.

**Cadangan: laptop + Cloudflare Tunnel.** Klik dua kali **`jalankan_demo.bat`**, lalu bagikan link `https://....trycloudflare.com` yang muncul. Link ini berganti setiap sesi.

Panduan lengkap, solusi error, dan tabel uji deployment ada di **`DEPLOY.md`**.
