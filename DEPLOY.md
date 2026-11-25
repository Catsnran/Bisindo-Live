# Panduan Deployment BISINDO Live

Sistem di-deploy sebagai **client–server** yang dapat diakses lewat **URL publik HTTPS**, sesuai proposal:

- **Server:** FastAPI + model SVM, dibungkus **Docker** (`Dockerfile`).
- **Hosting utama:** **Railway** (Trial: kredit $5 / 30 hari, tanpa kartu kredit) → link tetap `https://<nama>.up.railway.app`.
- **Cadangan demo:** server di laptop + **Cloudflare Tunnel** (`jalankan_demo.bat`) → link `https://<acak>.trycloudflare.com`.
- **Kode:** dikelola di **GitHub**.

```
Browser (webcam) --HTTPS, frame JPEG 320px, ±5 fps--> https://<nama>.up.railway.app
                                                       Docker: Uvicorn + FastAPI /predict
                 <-- JSON huruf + confidence --------  MediaPipe -> fitur -> scaler -> SVM
```

---

## 1. Persiapan (sekali saja)

### 1a. Model hasil notebook

Server membaca 5 file di `models/`:

- `best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `model_meta.json` — disimpan oleh **`BISINDO_Training.ipynb`** bagian 9 "Simpan Model Terbaik"
- `hand_landmarker.task` — diunduh oleh notebook bagian 2

Jalankan notebook sampai selesai bila file-file ini belum ada. Setiap kali notebook menghasilkan model baru, commit & push folder `models/` agar versi online ikut diperbarui.

### 1b. File deployment

| File | Fungsi |
|---|---|
| `Dockerfile` | Resep container: Python 3.12, library sistem OpenCV, menjalankan Uvicorn di port dari variabel `PORT` (bawaan 7860) |
| `requirements-deploy.txt` | Library server saja, versi dikunci sama dengan saat training (scikit-learn 1.9.1) |
| `.dockerignore` | Mencegah dataset, notebook, dan model eksperimen ikut masuk image |
| `.gitignore` | Mencegah `.venv`, `data/`, model ~70 MB, dan `cloudflared.exe` ikut ke GitHub |
| `jalankan_demo.bat` | Cadangan: server di laptop + Cloudflare Tunnel |

Isi container hanya: `src/config.py`, `src/landmark_extractor.py`, `backend/main.py`, `frontend/`, dan 5 file di `models/`.

---

## 2. Upload kode ke GitHub

Railway mengambil kode dari GitHub. Repo ini juga memenuhi poin "kode dikelola di GitHub" di proposal.

### Pilihan A — GitHub Desktop (tanpa perintah)

1. Buat akun di https://github.com bila belum punya.
2. Pasang **GitHub Desktop** (https://desktop.github.com), lalu login.
3. **File → Add local repository…** → pilih folder `D:\KULYAH\SEM5\ML\PROJECT BISINDO\bisindo_ml`.
4. Bila muncul *"This directory does not appear to be a Git repository"*, klik **create a repository** → **Create repository**.
5. Isi Summary `Versi awal BISINDO Live` → **Commit to main**.
6. **Publish repository**:
   - Nama: `bisindo-live`
   - Hapus centang **Keep this code private**
   - Klik **Publish repository**

### Pilihan B — git di terminal

Buat repo kosong di https://github.com/new, lalu dari folder `bisindo_ml`:

```powershell
git init
git add .
git commit -m "Versi awal BISINDO Live"
git branch -M main
git remote add origin https://github.com/<username>/bisindo-live.git
git push -u origin main
```

### Periksa repo

Pastikan di halaman repo GitHub ada:

- `Dockerfile` di root repo
- `models/best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `model_meta.json`, `hand_landmarker.task`

---

## 3. Deploy ke Railway (utama)

Trial Railway:

- **Kredit $5, berlaku 30 hari, tanpa kartu kredit.**
- 1 GB RAM per service. Server ini memakai ±300 MB.
- Service **tidak tidur** selama trial.
- Setelah trial habis, Free plan hanya $1/bulan dan 0,5 GB RAM. Itu cukup untuk beberapa hari pemakaian, setelah itu service berhenti sampai bulan berikutnya.

> **Atur waktu:** daftar Railway **beberapa hari sebelum presentasi**, supaya masa trial 30 hari mencakup hari presentasi dan masa penilaian.

### 3a. Daftar

1. Buka https://railway.com → **Login** → **Login with GitHub** (pakai akun GitHub yang sama dengan repo).
2. Railway memverifikasi akun secara otomatis berdasarkan umur & aktivitas akun GitHub.
   - Akun baru bisa mendapat **Limited Trial**, yang membatasi akses internet keluar. Server ini tetap jalan karena semua model sudah ada di dalam repo.

### 3b. Buat project dari repo GitHub

1. Di dashboard, klik **New Project** (atau **+ New**) → **Deploy from GitHub repo**.
2. Bila diminta, klik **Configure GitHub App** dan izinkan akses ke repo `bisindo-live`.
3. Pilih repo **`bisindo-live`** → **Deploy Now**.
4. Railway otomatis mendeteksi `Dockerfile` dan mulai build.

### 3c. Atur port & domain publik

1. Klik kotak service `bisindo-live` di canvas project.
2. Tab **Variables** → **New Variable**:
   - Name: `PORT`
   - Value: `7860`
   - Klik **Add**. Railway akan deploy ulang otomatis.
3. Tab **Settings** → bagian **Networking** → **Public Networking** → **Generate Domain**.
   - Bila diminta port, isi `7860`.
4. Domain berformat `https://<nama>.up.railway.app` akan muncul.
   - Bila ada ikon edit (pensil) di samping domain, nama depan bisa diganti, misalnya `bisindo-live`. Nama ini harus belum dipakai orang lain.
5. *(Opsional)* **Settings** → **Region**: pilih region Asia (mis. Singapore) agar latensi dari Indonesia lebih kecil, bila tersedia untuk akun trial.

### 3d. Tunggu build & buka

1. Tab **Deployments** → klik deployment terbaru → **Build Logs** (build pertama ±5–10 menit).
2. **Deploy Logs** harus menampilkan:
   ```
   [STARTUP] Model: SVM | kelas: 26
   Application startup complete.
   Uvicorn running on http://0.0.0.0:7860
   ```
3. Status berubah menjadi **Active / Success**.
4. Cek `https://<nama>.up.railway.app/health` → harus `"status": "ok"`.
5. Buka `https://<nama>.up.railway.app` → **Mulai Kamera** → **Allow**.

### 3e. Update & pantau

- **Update kode/model:** commit & push ke GitHub (GitHub Desktop: **Commit to main** → **Push origin**). Railway otomatis build ulang.
- **Sisa kredit:** menu akun → **Usage** / **Billing**. Server ±0,3 GB RAM memakan kira-kira $3–5 per bulan bila menyala terus, jadi kredit trial $5 cukup untuk masa trial.
- **Hemat kredit setelah presentasi:** service bisa dihentikan dari **Settings** (mis. *Remove/Sleep service*) bila sudah tidak dipakai.

### 3f. Bila build/deploy gagal

| Gejala | Penyebab & solusi |
|---|---|
| Build error saat `pip install` | Cek **Build Logs**; versi library di `requirements-deploy.txt` dikunci, kirim log error untuk diperbaiki |
| `File models/best_model.joblib tidak ditemukan` | Folder `models/` belum ter-push ke GitHub; commit & push ulang |
| Halaman *Application failed to respond* | Variabel `PORT` belum `7860` atau port domain bukan `7860` (langkah 3c) |
| Service berhenti / *out of credit* | Kredit trial habis; pakai cadangan Cloudflare Tunnel (bagian 4) |

---

## 4. Cadangan: laptop + Cloudflare Tunnel

Dipakai bila Railway bermasalah atau kredit habis. Gratis, tanpa akun, tanpa kartu kredit. Link acak dan berganti setiap sesi.

### 4a. Pasang cloudflared (sekali)

```powershell
winget install --id Cloudflare.cloudflared
```

Atau unduh `cloudflared-windows-amd64.exe` dari https://github.com/cloudflare/cloudflared/releases/latest, ganti nama menjadi `cloudflared.exe`, lalu taruh di folder `bisindo_ml`.

Python `.venv` juga harus sudah ada (dari folder `PROJECT BISINDO`: `python -m venv .venv` lalu `.venv\Scripts\pip install -r bisindo_ml\requirements.txt`).

### 4b. Jalankan

**Cara cepat:** klik dua kali **`jalankan_demo.bat`**. File ini:

1. Memeriksa `.venv` dan model.
2. Mencari/memasang cloudflared.
3. Menyalakan server di jendela **"BISINDO Server"** dan menunggu sampai siap.
4. Membuka http://localhost:8000.
5. Menampilkan link publik `https://....trycloudflare.com`.

Berhenti: **Ctrl+C** di jendela tunnel, lalu tutup jendela "BISINDO Server".

**Cara manual** (dua terminal, dari folder `bisindo_ml`, `.venv` aktif):

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Link hanya aktif selama laptop menyala, terhubung internet, dan kedua jendela terbuka. Matikan *sleep* laptop selama demo.

---

## 5. Uji Docker di laptop (opsional)

Butuh Docker Desktop:

```powershell
docker build -t bisindo-live .
docker run -p 7860:7860 bisindo-live
```

Buka http://localhost:7860.

---

## 6. Uji deployment (sesuai proposal)

| Pengujian | Cara | Target |
|---|---|---|
| Fungsional server | Buka `/health` dan `/docs` di URL Railway, coba `/predict/upload` dengan gambar dataset | `status: ok`, huruf benar |
| Fungsional client | Mulai kamera, peragakan beberapa huruf, cek label "Tangan tidak terdeteksi" | Huruf tampil stabil, tidak berkedip |
| Latensi end-to-end | Lihat kotak **latensi (ms)** di halaman; bandingkan `localhost` vs Railway | < 500 ms |
| Lintas browser | Chrome, Edge, Firefox | Kamera & prediksi berjalan |
| Lintas perangkat | Laptop + HP (Android/iPhone) lewat URL Railway | Kamera & prediksi berjalan |
| Privasi | Frame hanya diproses di memori; tidak ada kode yang menyimpan gambar | Tidak ada file tersimpan |

Catat rata-rata latensi (misalnya 20 pengamatan) untuk `localhost` dan Railway, serta hasil per browser/perangkat, dalam tabel untuk laporan.

## 7. Keterbatasan (untuk laporan)

- Hosting memakai Trial Railway, yang terbatas $5 / 30 hari. Setelah itu kuota Free plan ($1/bulan) hanya cukup untuk pemakaian singkat.
- Latensi dipengaruhi lokasi server dan kecepatan internet pengguna, karena setiap frame dikirim ke server.
- Cadangan Cloudflare Quick Tunnel bergantung pada laptop presenter dan memakai URL sementara yang berganti setiap sesi.
