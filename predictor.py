"""
Model inference — supports both PyTorch (.pt) and ONNX runtime.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True if HAS_TORCH else False
except Exception:
    HAS_ALBUMENTATIONS = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False


# ─── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SKIN_CLASSES = ["dry", "normal", "oily"]

SKIN_INFO: Dict[str, Dict] = {
    "dry": {
        "label": "Dry Skin",
        "emoji": "🌵",
        "color": "#f59e0b",
        "description": "Kulit kering cenderung terasa kencang, kasar, dan sering mengelupas. Produksi sebum rendah membuat kulit rentan terhadap iritasi.",
        "tips": [
            "Gunakan moisturizer krim atau balm yang kaya kandungan hyaluronic acid",
            "Cuci muka dengan cleanser gentle, hindari yang mengandung alkohol",
            "Aplikasikan serum vitamin E dan ceramide setiap malam",
            "Minum air minimal 2 liter per hari",
            "Hindari mandi air panas terlalu lama",
        ],
        "ingredients": ["Hyaluronic Acid", "Ceramide", "Shea Butter", "Vitamin E", "Glycerin"],
    },
    "normal": {
        "label": "Normal Skin",
        "emoji": "✨",
        "color": "#10b981",
        "description": "Kulit normal memiliki keseimbangan minyak dan kelembapan yang ideal. Pori-pori kecil, tekstur halus, dan jarang bermasalah.",
        "tips": [
            "Pertahankan rutinitas skincare yang konsisten",
            "Gunakan sunscreen SPF 30+ setiap pagi",
            "Lakukan eksfoliasi ringan 1-2x seminggu",
            "Pilih produk sesuai usia dan kondisi",
            "Jaga pola makan sehat dan tidur cukup",
        ],
        "ingredients": ["Niacinamide", "Vitamin C", "SPF", "AHA/BHA ringan", "Aloe Vera"],
    },
    "oily": {
        "label": "Oily Skin",
        "emoji": "💧",
        "color": "#3b82f6",
        "description": "Kulit berminyak memproduksi sebum berlebih, membuat wajah tampak mengkilap. Rentan terhadap komedo dan jerawat, namun lebih lambat berkerut.",
        "tips": [
            "Gunakan cleanser foaming atau gel yang mengandung salicylic acid",
            "Pilih moisturizer oil-free dan non-comedogenic",
            "Gunakan niacinamide untuk mengontrol produksi minyak",
            "Jangan skip moisturizer — kulit dehidrasi justru produksi minyak lebih banyak",
            "Eksfoliasi dengan BHA 2x seminggu",
        ],
        "ingredients": ["Salicylic Acid", "Niacinamide", "Zinc", "Tea Tree Oil", "BHA"],
    },
    "combination": {
        "label": "Combination Skin",
        "emoji": "🌗",
        "color": "#8b5cf6",
        "description": "Kulit kombinasi memiliki zona T (dahi, hidung, dagu) berminyak dan area pipi cenderung normal atau kering.",
        "tips": [
            "Gunakan produk berbeda untuk zona T dan zona pipi jika perlu",
            "Gel cleanser ringan cocok untuk semua area",
            "Aplikasikan moisturizer lebih banyak di area kering",
            "Clay mask di zona T 1x seminggu untuk kontrol minyak",
            "Pilih toner yang menyeimbangkan pH kulit",
        ],
        "ingredients": ["Niacinamide", "Hyaluronic Acid", "Clay", "Green Tea Extract", "BHA ringan"],
    },
}


# ─── Preprocessing ────────────────────────────────────────────────────────────
def get_val_transforms(size: int = 224):
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return None


def preprocess_image(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """Convert uploaded image bytes → normalized numpy array (1, 3, H, W)."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np  = np.array(pil_img)
    img_np  = cv2.resize(img_np, (size, size))

    if HAS_ALBUMENTATIONS:
        tfm    = get_val_transforms(size)
        tensor = tfm(image=img_np)["image"]         # (3, H, W) float32
        return tensor.unsqueeze(0).numpy()           # (1, 3, H, W)

    # Fallback: manual normalization
    img_f = img_np.astype(np.float32) / 255.0
    mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
    std   = np.array(IMAGENET_STD,  dtype=np.float32)
    img_f = (img_f - mean) / std
    return img_f.transpose(2, 0, 1)[None]           # (1, 3, H, W)


# ─── PyTorch model definition (must match training) ───────────────────────────
def _build_model(num_classes: int, backbone: str = "efficientnet_b0") -> nn.Module:
    if not HAS_TIMM:
        raise RuntimeError("timm is not installed. Run: pip install timm")

    class SkinClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = timm.create_model(backbone, pretrained=False, num_classes=0)
            feat_dim        = self.backbone.num_features
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Dropout(0.5),
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.backbone(x))

    return SkinClassifier()


# ─── Predictor class ──────────────────────────────────────────────────────────
class SkinPredictor:
    """Loads the trained model and runs inference on uploaded images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        onnx_path:  Optional[str] = None,
        skin_classes: List[str]   = None,
        image_size: int           = 224,
        device: str               = "cpu",
    ):
        self.image_size   = image_size
        self.device       = device
        self.skin_classes = skin_classes or SKIN_CLASSES
        self.num_classes  = len(self.skin_classes)
        self._session     = None   # ONNX
        self._model       = None   # PyTorch

        # Prefer ONNX (faster CPU inference), fallback to PyTorch
        if onnx_path and Path(onnx_path).exists() and HAS_ONNX:
            self._load_onnx(onnx_path)
        elif model_path and model_path and Path(model_path).exists() and HAS_TORCH:
            self._load_pytorch(model_path)
        elif not HAS_ONNX:
            raise RuntimeError("onnxruntime tidak terinstall. Jalankan: pip install onnxruntime")
        else:
            raise FileNotFoundError(
                f"Model tidak ditemukan. Dicek:\n  ONNX : {onnx_path}\n  PT   : {model_path}"
            )

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_onnx(self, path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(path, providers=providers)
        self._mode    = "onnx"

        # Deteksi jumlah kelas dari output shape model
        out_shape = self._session.get_outputs()[0].shape
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            n = out_shape[1]
            if n != self.num_classes:
                self.num_classes  = n
                self.skin_classes = SKIN_CLASSES[:n]
                print(f"[SkinPredictor] Model memiliki {n} kelas: {self.skin_classes}")

        print(f"[SkinPredictor] Loaded ONNX model: {path}")

    def _load_pytorch(self, path: str):
        device = torch.device(self.device)
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        state  = ckpt.get("model_state_dict", ckpt)
        for k, v in state.items():
            if k.endswith("classifier.5.weight") or k.endswith("classifier.3.weight"):
                self.num_classes  = v.shape[0]
                self.skin_classes = SKIN_CLASSES[: self.num_classes]
                break

        self._model = _build_model(self.num_classes)
        self._model.load_state_dict(state, strict=False)
        self._model.to(device).eval()
        self._mode  = "pytorch"
        print(f"[SkinPredictor] Loaded PyTorch model: {path} ({self.num_classes} classes)")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, image_bytes: bytes) -> Dict:
        tensor = preprocess_image(image_bytes, self.image_size)  # (1,3,H,W) float32

        if self._mode == "onnx":
            inp_name = self._session.get_inputs()[0].name
            logits   = self._session.run(None, {inp_name: tensor})[0]          # (1, C)
        else:
            with torch.no_grad():
                t      = torch.from_numpy(tensor).to(torch.device(self.device))
                logits = self._model(t).cpu().numpy()

        probs   = self._softmax(logits[0])                                      # (C,)
        top_idx = int(probs.argmax())
        label   = self.skin_classes[top_idx]

        info = SKIN_INFO.get(label, {
            "label": label.capitalize(),
            "emoji": "🔍",
            "color": "#6b7280",
            "description": "",
            "tips": [],
            "ingredients": [],
        })

        return {
            "skin_type":     label,
            "label":         info["label"],
            "emoji":         info["emoji"],
            "color":         info["color"],
            "confidence":    round(float(probs[top_idx]) * 100, 1),
            "probabilities": {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(self.skin_classes, probs)
            },
            "description": info["description"],
            "tips":        info["tips"],
            "ingredients": info["ingredients"],
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
