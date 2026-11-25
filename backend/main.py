"""
main.py — BAGIAN 3: Backend REST API (FastAPI) untuk klasifikasi huruf BISINDO.

Saat startup server memuat:
  - models/best_model.joblib     (classifier terbaik hasil BISINDO_Training.ipynb)
  - models/scaler.joblib         (StandardScaler yang di-fit di data latih)
  - models/label_encoder.joblib  (indeks kelas <-> huruf A..Z)
  - models/hand_landmarker.task  (MediaPipe)

Endpoint:
  GET  /health          status server & info model
  POST /predict         JSON {"image": "<base64 JPEG / data URL>"}  -> dipakai frontend
  POST /predict/upload  multipart file gambar                       -> untuk uji via Swagger/curl
  GET  /                frontend web (Bagian 4), bila folder frontend/ sudah ada

Ekstraksi fitur memakai modul yang SAMA dengan training (src/landmark_extractor.py),
sehingga normalisasi, urutan tangan, dan zero-padding dijamin identik.

Menjalankan (dari folder bisindo_ml):
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
lalu buka http://localhost:8000/docs
"""
from __future__ import annotations

import base64
import binascii
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Agar modul di src/ (config, landmark_extractor) bisa di-import
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (  # noqa: E402
    BEST_MODEL_PATH,
    LABEL_ENCODER_PATH,
    META_PATH,
    NUM_FEATURES,
    REALTIME_VARIANTS,
    SCALER_PATH,
)
from landmark_extractor import HandLandmarkExtractor  # noqa: E402

MAX_IMAGE_SIDE = 640          # frame lebih besar di-resize agar inferensi cepat
MAX_PAYLOAD_BYTES = 5 * 1024 * 1024
TOP_K = 3
FRONTEND_DIR = ROOT_DIR / "frontend"


# ---------------------------------------------------------------------------
# Skema request / response (otomatis muncul di dokumentasi /docs)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str = Field(..., description="Gambar JPEG/PNG dalam base64 (boleh dengan prefix data:image/...;base64,)")


class TopKItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    hand_detected: bool
    num_hands: int
    prediction: str | None = None
    confidence: float | None = None
    top_k: list[TopKItem] = []
    landmarks: list[list[list[float]]] = Field(
        default=[], description="Per tangan: 21 titik [x, y] ternormalisasi 0..1 (untuk digambar di frontend)"
    )
    inference_ms: float


# ---------------------------------------------------------------------------
# State model (dimuat sekali saat startup)
# ---------------------------------------------------------------------------
class ModelState:
    model = None
    scaler = None
    label_encoder = None
    extractor: HandLandmarkExtractor | None = None
    meta: dict = {}
    # MediaPipe HandLandmarker tidak thread-safe, sedangkan endpoint sync FastAPI
    # dijalankan di thread pool -> akses detektor dikunci.
    lock = threading.Lock()


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for p in (BEST_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH):
        if not p.exists():
            raise RuntimeError(f"File {p} tidak ditemukan. Jalankan dulu notebook BISINDO_Training.ipynb sampai bagian 9 (Simpan Model)")
    state.model = joblib.load(BEST_MODEL_PATH)
    state.scaler = joblib.load(SCALER_PATH)
    state.label_encoder = joblib.load(LABEL_ENCODER_PATH)
    state.meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    state.extractor = HandLandmarkExtractor()
    print(f"[STARTUP] Model: {state.meta.get('best_model', type(state.model).__name__)} | "
          f"kelas: {len(state.label_encoder.classes_)}")
    print("[STARTUP] Lokal: buka http://localhost:<port> (bukan 0.0.0.0; kamera butuh localhost/HTTPS)")
    yield
    state.extractor.close()


app = FastAPI(
    title="BISINDO Sign Language Classifier API",
    description="Klasifikasi huruf BISINDO (A-Z) dari landmark tangan MediaPipe.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: izinkan frontend dari origin lain (mis. dibuka via Live Server / file lain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def decode_base64_image(data: str) -> np.ndarray:
    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]              # buang prefix data URL
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "String base64 tidak valid")
    return decode_bytes(raw)


def decode_bytes(raw: bytes) -> np.ndarray:
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise HTTPException(413, "Ukuran gambar terlalu besar (maks 5 MB)")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Data bukan gambar yang valid (JPEG/PNG)")
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def predict_image(img: np.ndarray) -> PredictResponse:
    t0 = time.perf_counter()

    # 1) Ekstraksi landmark + normalisasi (logika identik dengan training)
    with state.lock:
        res = state.extractor.extract(img, variants=REALTIME_VARIANTS)

    landmarks = [lm[:, :2].astype(float).round(4).tolist() for lm in res.raw_landmarks]
    if res.num_hands == 0:
        return PredictResponse(hand_detected=False, num_hands=0,
                               inference_ms=round(1000 * (time.perf_counter() - t0), 1))

    # 2) Scaling -> 3) Prediksi probabilitas
    x = state.scaler.transform(res.features.reshape(1, NUM_FEATURES))
    proba = state.model.predict_proba(x)[0]
    classes = state.label_encoder.inverse_transform(state.model.classes_)
    order = np.argsort(proba)[::-1][:TOP_K]
    top_k = [TopKItem(label=str(classes[i]), confidence=round(float(proba[i]), 4)) for i in order]

    return PredictResponse(
        hand_detected=True,
        num_hands=res.num_hands,
        prediction=top_k[0].label,
        confidence=top_k[0].confidence,
        top_k=top_k,
        landmarks=landmarks,
        inference_ms=round(1000 * (time.perf_counter() - t0), 1),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if state.model is not None else "loading",
        "model": state.meta.get("best_model", type(state.model).__name__ if state.model else None),
        "classes": list(state.label_encoder.classes_) if state.label_encoder is not None else [],
        "test_metrics": state.meta.get("test_metrics"),
        "trained_at": state.meta.get("trained_at"),
    }


# Endpoint didefinisikan sebagai `def` (bukan async) karena MediaPipe & sklearn
# bersifat CPU-bound; FastAPI menjalankannya di thread pool sehingga event loop
# tidak terblokir.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return predict_image(decode_base64_image(req.image))


@app.post("/predict/upload", response_model=PredictResponse)
def predict_upload(file: UploadFile = File(...)):
    return predict_image(decode_bytes(file.file.read()))


# Frontend (Bagian 4) disajikan dari server yang sama -> tidak perlu urus CORS
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")
