# Panduan Deployment BISINDO Live

Sesuai proposal (bagian *Deployment dan Program Client*):

- Server dibungkus **Docker**
- Dipublikasikan ke **Render** (paket Free, tanpa kartu kredit) sehingga dapat diakses lewat URL publik
- Kode dikelola di **GitHub**
- **Uvicorn lokal** sebagai cadangan saat demonstrasi

```
Browser (webcam) --frame JPEG 320px, ±5 fps--> FastAPI /predict (Docker @ Render)
                 <--JSON huruf + confidence--   MediaPipe -> normalisasi -> scaler -> SVM
```

## File deployment

File di folder `bisindo_ml`:

| File | Fungsi |
|---|---|
| `Dockerfile` | Resep container: Python 3.12, library sistem OpenCV, menjalankan server Uvicorn (membaca port dari variabel `PORT`) |
| `requirements-deploy.txt` | Library server saja, versi dikunci sama dengan saat training (scikit-learn 1.9.1) |
| `.dockerignore` | Mencegah dataset, notebook, dan model eksperimen ikut masuk image |
| `.gitignore` | Mencegah `.venv`, `data/`, dan model ~70 MB ikut ke GitHub |

Isi container hanya file yang dibutuhkan untuk prediksi:

- `src/config.py`
- `src/landmark_extractor.py`
- `backend/main.py`
- `frontend/`
- 5 file di `models/`: `best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `model_meta.json`, `hand_landmarker.task`

> **Sumber model:** kelima file di `models/` dihasilkan oleh notebook **`BISINDO_Training.ipynb`**.
> - Empat file pertama disimpan di bagian 9 "Simpan Model Terbaik".
> - `hand_landmarker.task` diunduh di bagian 2.
>
> Jalankan notebook sampai selesai **sebelum** deploy. Setiap kali notebook menghasilkan model baru, commit & push folder `models/` agar versi online ikut diperbarui.

---

## 1. Upload kode ke GitHub

Render mengambil kode dari GitHub. Repo ini juga memenuhi poin "kode dikelola di GitHub" di proposal.

### Pilihan A — GitHub Desktop (tanpa perintah)

1. Buat akun di https://github.com bila belum punya.
2. Unduh & pasang **GitHub Desktop** (https://desktop.github.com), lalu login dengan akun GitHub.
3. **File → Add local repository…** → pilih folder `D:\KULYAH\SEM5\ML\PROJECT BISINDO\bisindo_ml`.
4. Akan muncul tulisan *"This directory does not appear to be a Git repository"*. Klik **create a repository**, lalu **Create repository**.
5. Di kolom kiri bawah, isi Summary `Versi awal BISINDO Live`, lalu klik **Commit to main**.
6. Klik **Publish repository**:
   - Nama: `bisindo-live`
   - **Hapus centang "Keep this code private"**
   - Klik **Publish repository**.

### Pilihan B — git di terminal

Buat repo kosong dulu di https://github.com/new, lalu jalankan dari folder `bisindo_ml`:

```powershell
git init
git add .
git commit -m "Versi awal BISINDO Live"
git branch -M main
git remote add origin https://github.com/<username>/bisindo-live.git
git push -u origin main
```

### Periksa repo

`.gitignore` otomatis mengecualikan `data/` dan `models/all/`. `RandomForest_pipeline.joblib` (~70 MB) melewati batas GitHub, jadi tidak ikut.

Pastikan di halaman repo GitHub file berikut ada:

- `Dockerfile` (di root repo)
- `models/best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `model_meta.json`, `hand_landmarker.task` (hasil notebook; jika belum ada, jalankan `BISINDO_Training.ipynb` dulu)

---

## 2. Deploy ke Render

Keterbatasan:

- CPU hanya 0,1, sehingga satu prediksi butuh ±0,5–1 detik. Prediksi tetap jalan, hanya lebih lambat dari target 500 ms.
- Service **tidur setelah 15 menit tanpa pengunjung**. Saat dibuka lagi, butuh ±1 menit untuk bangun.

### 2a. Buat Web Service

1. Daftar di https://render.com → **Get Started** → pilih **GitHub** (login pakai akun GitHub).
2. Di Dashboard, klik **+ New** (kanan atas) → **Web Service**.
3. Pilih **GitHub**. Saat diminta, izinkan Render mengakses repo `bisindo-live`. Pilih repo itu, lalu klik **Connect**.
4. Isi form:
   - **Name:** `bisindo-live` (jadi bagian dari URL)
   - **Region:** **Singapore** (terdekat dari Indonesia, latensi lebih kecil)
   - **Branch:** `main`
   - **Root Directory:** kosongkan
   - **Language:** **Docker** (biasanya terdeteksi otomatis karena ada `Dockerfile`)
   - **Instance Type:** **Free**
   - **Environment Variables:** tidak perlu diisi. Render mengisi `PORT` sendiri, dan `Dockerfile` sudah membacanya.
5. Klik **Deploy Web Service**.

### 2b. Tunggu build & buka

1. Halaman service menampilkan **Logs**. Build pertama butuh ±5–10 menit.
2. Build berhasil bila log menampilkan `[STARTUP] Model: SVM | kelas: 26`, lalu `Application startup complete`, dan status berubah menjadi **Live**.
3. URL ada di bagian atas halaman service, formatnya `https://bisindo-live.onrender.com`. Bisa ada akhiran acak bila nama sudah dipakai orang lain.
4. Cek server di `https://<url>/health`. Hasilnya harus `"status": "ok"`.
5. Buka URL utama → **Mulai Kamera** → **Allow**.

### 2c. Update setelah mengubah kode atau model

Commit & push ke GitHub (GitHub Desktop: **Commit to main** → **Push origin**). Render otomatis build ulang (*Auto-Deploy*).

### 2d. Tips sebelum presentasi

- Buka URL Render **±2 menit sebelum presentasi** supaya service sudah bangun.
- Untuk demo real-time yang mulus, pakai laptop + Cloudflare Tunnel (bagian 3), lalu tunjukkan link Render sebagai bukti deployment publik. Ini sesuai proposal ("Uvicorn lokal sebagai cadangan saat demonstrasi").
- Catat perbedaan latensi lokal vs Render di laporan sebagai hasil "uji deployment".

---

## 3. Cadangan saat demo: Uvicorn lokal

Jalankan server dari folder `bisindo_ml`:

```powershell
cd bisindo_ml
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Buka **http://localhost:8000**.

Agar bisa diakses perangkat lain lewat HTTPS (syarat kamera di browser), jalankan di terminal kedua:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Bagikan link `https://....trycloudflare.com` yang muncul. Link hanya aktif selama laptop menyala dan kedua terminal tetap berjalan.

Uji Docker di laptop (opsional, butuh Docker Desktop):

```powershell
docker build -t bisindo-live .
docker run -p 7860:7860 bisindo-live      # buka http://localhost:7860
```

---

## 4. Uji deployment (sesuai proposal)

| Pengujian | Cara | Target |
|---|---|---|
| Fungsional server | Buka `/health` dan `/docs`, coba `/predict/upload` dengan gambar dataset | `status: ok`, huruf benar |
| Fungsional client | Mulai kamera, peragakan beberapa huruf, cek label "Tangan tidak terdeteksi" | Huruf tampil stabil, tidak berkedip |
| Latensi end-to-end | Lihat kotak **latensi (ms)** di halaman (waktu kirim frame → terima hasil) | < 500 ms |
| Lintas browser | Chrome, Edge, Firefox (laptop) | Kamera & prediksi berjalan |
| Lintas perangkat | Laptop + HP (Android/iPhone) lewat URL publik | Kamera & prediksi berjalan |
| Privasi | Frame hanya diproses di memori; tidak ada kode yang menyimpan gambar di server | Tidak ada file tersimpan |

Catat hasil latensi (misalnya rata-rata 20 pengamatan) dan hasil per browser/perangkat dalam tabel untuk laporan.
