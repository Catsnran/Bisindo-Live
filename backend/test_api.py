"""
test_api.py — Uji cepat backend tanpa frontend.

Pastikan server sudah jalan (uvicorn backend.main:app --port 8000), lalu:
    python backend/test_api.py                                # pakai 1 gambar per huruf dari dataset
    python backend/test_api.py --image "path/ke/gambar.jpg"   # gambar tertentu
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from config import DATASET_DIR  # noqa: E402


def call(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--image", type=Path)
    args = ap.parse_args()

    print("GET /health ->", json.dumps(call(f"{args.url}/health"), indent=2))

    if args.image:
        images = [(args.image, "?")]
    else:
        rng = random.Random(0)
        images = []
        for d in sorted((DATASET_DIR / "val").iterdir()):
            files = [f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if files:
                images.append((rng.choice(files), d.name))

    correct = 0
    for path, label in images:
        b64 = base64.b64encode(path.read_bytes()).decode()
        r = call(f"{args.url}/predict", {"image": b64})
        pred = r["prediction"] or "-"
        ok = pred == label
        correct += ok
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "-"
        print(f"{label:>2} -> {pred:>2}  conf={conf}  tangan={r['num_hands']}  "
              f"{r['inference_ms']:6.1f} ms  {'OK' if ok else ''}  {path.name}")
    if not args.image:
        print(f"\nBenar {correct}/{len(images)}")


if __name__ == "__main__":
    main()
