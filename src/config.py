"""
config.py — Konfigurasi terpusat proyek BISINDO.

Semua path & konstanta ada di sini supaya skrip lain (ekstraksi, training,
backend FastAPI) memakai nilai yang sama persis.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
# Folder root kode: .../PROJECT BISINDO/bisindo_ml
ROOT_DIR = Path(__file__).resolve().parent.parent
# Dataset ada satu level di atas: .../PROJECT BISINDO/bisindo/images/{train,val}/A..Z
DATASET_DIR = ROOT_DIR.parent / "bisindo" / "images"

DATA_DIR = ROOT_DIR / "data"            # hasil ekstraksi fitur (CSV)
MODELS_DIR = ROOT_DIR / "models"        # model .joblib + model MediaPipe .task
REPORTS_DIR = ROOT_DIR / "reports"      # grafik, confusion matrix, metrik

FEATURES_CSV = DATA_DIR / "landmarks.csv"
FAILED_CSV = DATA_DIR / "failed_images.csv"

# Model MediaPipe Hand Landmarker (Tasks API). Diunduh otomatis bila belum ada.
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Artefak model terbaik (dipakai backend)
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
META_PATH = MODELS_DIR / "model_meta.json"

# ---------------------------------------------------------------------------
# Fitur
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21            # titik per tangan
NUM_COORDS = 3                # x, y, z
MAX_HANDS = 2
FEATURES_PER_HAND = NUM_LANDMARKS * NUM_COORDS        # 63
NUM_FEATURES = FEATURES_PER_HAND * MAX_HANDS          # 126

FEATURE_COLUMNS = [
    f"h{h}_p{p}_{c}"
    for h in range(MAX_HANDS)
    for p in range(NUM_LANDMARKS)
    for c in ("x", "y", "z")
]

# Ambang deteksi MediaPipe. Untuk gambar statis (dataset) sedikit lebih
# longgar agar lebih banyak gambar terdeteksi.
MIN_DETECTION_CONF = 0.3
MIN_PRESENCE_CONF = 0.3

# Urutan preprocessing yang dicoba per gambar (lihat landmark_extractor.PREPROCESSORS).
# Dataset: median7 dulu (paling efektif untuk gambar ber-noise), lalu varian lain
# bila belum ditemukan 2 tangan.
DATASET_VARIANTS = ("median7", "orig", "median5", "median9")
# Real-time (backend): cukup 2 varian agar latensi rendah.
REALTIME_VARIANTS = ("median7", "orig")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1_macro"          # metrik utama pemilihan hyperparameter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
