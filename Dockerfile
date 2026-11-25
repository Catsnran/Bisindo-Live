# Dockerfile — server BISINDO Live (FastAPI + frontend) dalam container (opsional, siap dipindah ke hosting).
# Build & jalankan lokal:
#   docker build -t bisindo-live .
#   docker run -p 7860:7860 bisindo-live      -> buka http://localhost:7860
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Library sistem: OpenCV (libGL, glib), MediaPipe Tasks (EGL, GLES) & audio (portaudio)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 libegl1 libgles2 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Jalankan server sebagai user biasa (bukan root) demi keamanan
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# Cek saat build: semua library native MediaPipe harus bisa dimuat.
# Bila ada yang kurang, build gagal di sini dan Build Logs menampilkan daftarnya.
RUN python -c "import glob,os,subprocess,mediapipe as m; libs=glob.glob(os.path.dirname(m.__file__)+'/**/*.so*',recursive=True); miss=sorted({l.split()[0] for f in libs for l in subprocess.run(['ldd',f],capture_output=True,text=True).stdout.splitlines() if 'not found' in l}); print('MediaPipe native libs missing:', miss or 'none'); assert not miss"

# Hanya file yang dibutuhkan saat inferensi (dataset, notebook, & hasil eksperimen tidak ikut)
COPY --chown=user src/config.py src/landmark_extractor.py src/
COPY --chown=user backend/main.py backend/
COPY --chown=user frontend/ frontend/
COPY --chown=user models/best_model.joblib models/scaler.joblib models/label_encoder.joblib \
                  models/model_meta.json models/hand_landmarker.task models/

USER user

# Port bisa diatur lewat variabel PORT (bawaan 7860)
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
