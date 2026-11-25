"""
landmark_extractor.py — Modul inti ekstraksi & normalisasi landmark tangan.

Modul ini SENGAJA dipisah supaya dipakai oleh DUA tempat dengan logika identik:
  1. BISINDO_Training.ipynb -> saat membuat dataset fitur untuk training (logika disalin identik)
  2. backend FastAPI       -> saat memprediksi frame dari kamera
Kalau logikanya berbeda sedikit saja (urutan tangan, cara normalisasi),
akurasi saat deployment akan turun drastis (training-serving skew).

Alur per gambar:
  BGR image -> (median blur untuk hapus noise) -> RGB -> MediaPipe HandLandmarker (maks 2 tangan)
  -> tiap tangan: koordinat piksel (x*w, y*h, z*w)
  -> normalisasi relatif wrist (titik 0) + skala
  -> urutkan tangan (kiri-ke-kanan di gambar) -> zero-padding bila 1 tangan
  -> vektor 126 fitur
"""
from __future__ import annotations

import os

# Kurangi log C++ MediaPipe/LiteRT yang ramai di terminal (harus sebelum import mediapipe)
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    DATASET_VARIANTS,
    FEATURES_PER_HAND,
    HAND_MODEL_PATH,
    HAND_MODEL_URL,
    MAX_HANDS,
    MIN_DETECTION_CONF,
    MIN_PRESENCE_CONF,
    NUM_FEATURES,
    NUM_LANDMARKS,
)

WRIST = 0


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def ensure_hand_model(path: Path = HAND_MODEL_PATH) -> Path:
    """Unduh model hand_landmarker.task (~7.8 MB) bila belum ada."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Mengunduh model MediaPipe ke {path} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def read_image(path: str | Path) -> np.ndarray | None:
    """Baca gambar dengan aman (mendukung path Windows berisi spasi/unicode)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR


# ---------------------------------------------------------------------------
# Varian preprocessing citra sebelum deteksi
# ---------------------------------------------------------------------------
# Banyak gambar `augmented_image_*` di dataset diberi noise (bintik-bintik)
# sehingga detektor telapak MediaPipe gagal total. Median blur menghapus noise
# tersebut tanpa merusak tepi jari. Hasil uji pada 31 gambar yang gagal:
#   asli: 1/31 terdeteksi  |  median 5: 31/31  |  median 7: 31/31 (+ 2 tangan lebih lengkap)
# Pada gambar bersih, median blur tidak mengubah hasil deteksi.
PREPROCESSORS = {
    "orig": lambda img: img,
    "median5": lambda img: cv2.medianBlur(img, 5),
    "median7": lambda img: cv2.medianBlur(img, 7),
    "median9": lambda img: cv2.medianBlur(img, 9),
}


def normalize_hand(points: np.ndarray) -> np.ndarray:
    """
    Normalisasi satu tangan.

    points : array (21, 3) dalam satuan piksel.
    1) Translasi : kurangi koordinat wrist -> invarian terhadap POSISI tangan.
    2) Skala     : bagi dengan jarak terjauh titik mana pun ke wrist (bidang x-y)
                   -> invarian terhadap UKURAN tangan / jarak ke kamera.
    Hasil: array (63,) dengan nilai kira-kira di rentang [-1, 1].
    """
    rel = points - points[WRIST]
    scale = np.max(np.linalg.norm(rel[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    return (rel / scale).astype(np.float32).flatten()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    features: np.ndarray          # (126,)
    num_hands: int                # 0, 1, atau 2
    handedness: list[str]         # mis. ["Left", "Right"] (sesuai urutan slot)
    raw_landmarks: list[np.ndarray]  # landmark ternormalisasi MediaPipe (untuk visualisasi)
    variant: str = "orig"         # varian preprocessing yang menghasilkan deteksi ini


class HandLandmarkExtractor:
    """
    Wrapper MediaPipe HandLandmarker (Tasks API, mediapipe >= 0.10).

    Catatan: API lama `mp.solutions.hands` sudah dihapus pada versi mediapipe
    terbaru, jadi di sini dipakai Tasks API yang didukung jangka panjang.
    """

    def __init__(
        self,
        model_path: Path = HAND_MODEL_PATH,
        min_detection_conf: float = MIN_DETECTION_CONF,
        min_presence_conf: float = MIN_PRESENCE_CONF,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_hand_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,   # tiap gambar diproses independen
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_presence_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    # -- context manager agar resource MediaPipe dilepas dengan rapi --------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # -----------------------------------------------------------------------
    def extract(self, image_bgr: np.ndarray, variants=DATASET_VARIANTS) -> ExtractionResult:
        """
        Ekstrak vektor 126 fitur dari satu gambar BGR.

        `variants` = urutan preprocessing yang dicoba (lihat PREPROCESSORS).
        Dipilih hasil dengan jumlah tangan terbanyak; berhenti lebih awal begitu
        2 tangan ditemukan. Untuk real-time cukup 1-2 varian agar cepat.
        """
        best = None
        for name in variants:
            res = self._extract_single(PREPROCESSORS[name](image_bgr))
            res.variant = name
            if best is None or res.num_hands > best.num_hands:
                best = res
            if best.num_hands >= MAX_HANDS:
                break
        return best

    def _extract_single(self, image_bgr: np.ndarray) -> ExtractionResult:
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hands = []
        for i, lms in enumerate(result.hand_landmarks[:MAX_HANDS]):
            # Koordinat MediaPipe dinormalisasi 0..1 terhadap lebar & tinggi
            # gambar. Dikonversi ke piksel agar rasio aspek tidak terdistorsi.
            # z memakai skala yang kira-kira sama dengan x (dokumentasi MediaPipe).
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float32)
            label = (
                result.handedness[i][0].category_name
                if result.handedness and i < len(result.handedness)
                else "Unknown"
            )
            raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            hands.append((pts, label, raw))

        return build_feature_vector(hands)


def build_feature_vector(hands: list[tuple[np.ndarray, str, np.ndarray]]) -> ExtractionResult:
    """
    Susun vektor 126 fitur dari daftar tangan terdeteksi.

    Urutan tangan dibuat KONSISTEN:
      - 2 tangan  -> diurutkan berdasarkan posisi x wrist di gambar
                     (tangan paling kiri di gambar = slot 0).
                     Label handedness MediaPipe tidak dipakai untuk urutan karena
                     tidak stabil pada gambar hasil flip/augmentasi.
      - 1 tangan  -> masuk slot 0, slot 1 diisi nol (zero-padding).
      - 0 tangan  -> semua nol, num_hands = 0 (pemanggil sebaiknya membuang).
    """
    hands = sorted(hands, key=lambda t: t[0][WRIST, 0])

    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    for slot, (pts, _, _) in enumerate(hands[:MAX_HANDS]):
        assert pts.shape == (NUM_LANDMARKS, 3)
        start = slot * FEATURES_PER_HAND
        features[start:start + FEATURES_PER_HAND] = normalize_hand(pts)

    return ExtractionResult(
        features=features,
        num_hands=len(hands),
        handedness=[t[1] for t in hands],
        raw_landmarks=[t[2] for t in hands],
    )
