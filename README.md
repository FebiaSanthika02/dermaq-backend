# Backend — Dermiq AI Skin Analysis

## Setup & Jalankan

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Model akan **otomatis didownload** dari Hugging Face saat pertama kali dijalankan.
Tidak perlu upload manual.

## Endpoint

| Method | Path       | Keterangan                       |
|--------|------------|----------------------------------|
| GET    | /          | Info API                         |
| GET    | /health    | Status model                     |
| POST   | /analyze   | Upload gambar → hasil analisis   |

Dokumentasi lengkap: http://localhost:8000/docs

## Model

Tersimpan di: https://huggingface.co/febiasanthika/skin-analysis
- `skin_classifier.onnx` — Klasifikasi jenis kulit
- `acne_detector.onnx`   — Deteksi jerawat (opsional)
