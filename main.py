"""
FastAPI backend for AI Skin Analysis — Dermiq
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from predictor import SkinPredictor

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS_DIR  = Path(os.getenv("MODELS_DIR", "models"))
ONNX_PATH   = str(MODELS_DIR / "skin_classifier.onnx")
MODEL_PATH  = str(MODELS_DIR / "best_model.pt")
IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE", "224"))
DEVICE      = os.getenv("DEVICE", "cpu")
MAX_MB      = float(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

# Hugging Face model URLs
HF_ONNX_URL = os.getenv(
    "HF_ONNX_URL",
    "https://huggingface.co/febiasanthika/skin-analysis/resolve/main/skin_classifier.onnx",
)
HF_ACNE_URL = os.getenv(
    "HF_ACNE_URL",
    "https://huggingface.co/febiasanthika/skin-analysis/resolve/main/acne_detector.onnx",
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


# ─── Auto-download model dari Hugging Face ────────────────────────────────────
def download_file(url: str, dest: Path):
    """Download file dari URL ke dest dengan progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} dari Hugging Face...")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            print(f"\r  {dest.name}: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  {dest.name}: selesai ({dest.stat().st_size / 1e6:.1f} MB)")


def ensure_models():
    """Pastikan model tersedia. Download dari HF jika belum ada."""
    onnx_path = Path(ONNX_PATH)

    if not onnx_path.exists():
        print("Model belum ada — mengunduh dari Hugging Face...")
        try:
            download_file(HF_ONNX_URL, onnx_path)
        except Exception as e:
            print(f"  Gagal download skin_classifier.onnx: {e}")

    # Opsional: download acne detector
    acne_path = MODELS_DIR / "acne_detector.onnx"
    if not acne_path.exists():
        try:
            download_file(HF_ACNE_URL, acne_path)
        except Exception as e:
            print(f"  Gagal download acne_detector.onnx: {e}")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Dermiq — AI Skin Analysis API",
    description = "Analisis jenis kulit dari foto wajah",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Load model saat startup ──────────────────────────────────────────────────
predictor: Optional[SkinPredictor] = None

@app.on_event("startup")
async def startup():
    global predictor
    print("=" * 50)
    print("  Dermiq Backend — Memuat model...")
    print("=" * 50)

    # 1. Download model jika belum ada
    ensure_models()

    # 2. Load ke memori
    try:
        predictor = SkinPredictor(
            model_path = MODEL_PATH if Path(MODEL_PATH).exists() else None,
            onnx_path  = ONNX_PATH,
            image_size = IMAGE_SIZE,
            device     = DEVICE,
        )
        print("Model berhasil dimuat. Backend siap!")
    except FileNotFoundError as e:
        print(f"Model tidak ditemukan: {e}")
        print("Backend tetap berjalan tapi /analyze tidak tersedia.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")


# ─── Response schema ──────────────────────────────────────────────────────────
class AnalysisResponse(BaseModel):
    skin_type:     str
    label:         str
    emoji:         str
    color:         str
    confidence:    float
    probabilities: dict
    description:   str
    tips:          list
    ingredients:   list
    latency_ms:    float


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Dermiq API berjalan",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": predictor is not None,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model belum tersedia. Tunggu sebentar dan coba lagi.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code = 400,
            detail      = f"Format '{file.content_type}' tidak didukung. Gunakan JPG, PNG, atau WebP.",
        )

    contents = await file.read()
    if len(contents) / (1024 * 1024) > MAX_MB:
        raise HTTPException(
            status_code = 413,
            detail      = f"File terlalu besar. Maksimum {MAX_MB} MB.",
        )

    try:
        t0     = time.perf_counter()
        result = predictor.predict(contents)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")

    return AnalysisResponse(**result, latency_ms=round(elapsed_ms, 1))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code = 500,
        content     = {"detail": f"Terjadi kesalahan: {str(exc)}"},
    )
