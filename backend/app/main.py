"""
TruCast / DeepShield — FastAPI Backend v2.3
Fixes:
  - /liveness/check (single frame, useless) REPLACED by /liveness/verify (multi-frame, real analysis)
  - mediapipe AttributeError on Python 3.11/Windows (lazy init kept, but liveness no longer depends on it)
  - Liveness verdict now comes from actual frame-difference analysis, NOT deepfake probability

Run (from backend/ folder):
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"""

import cv2
import numpy as np
import base64
import json
import random
import os
import glob
from pathlib import Path
from datetime import datetime
from liveness_verifier import router as liveness_router
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Liveness multi-frame router (the fix) ──────────────────────────────────────
from liveness_verifier import router as liveness_router

# ── Crypto ─────────────────────────────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    _CRYPTO_OK = True
except ImportError:
    _CRYPTO_OK = False
    print("⚠  cryptography not installed — /crypto/* endpoints disabled")

# ── mediapipe: lazy import (safe on Python 3.11 / Windows) ─────────────────────
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("⚠  mediapipe not installed — /liveness/check fallback disabled")

# ─────────────────────────────────────────────
#  Path Resolution
# ─────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_ROOT    = _BACKEND.parent

ONNX_CANDIDATES = [
    _BACKEND / "models" / "deepshield.onnx",
    _HERE    / "models" / "deepshield.onnx",
    _ROOT    / "models" / "deepshield.onnx",
    *[Path(p) for p in glob.glob(str(_ROOT / "**" / "deepshield.onnx"), recursive=True)],
    *[Path(p) for p in glob.glob(str(_ROOT / "**" / "*.onnx"),          recursive=True)],
]

PT_CANDIDATES = [
    *[Path(p) for p in glob.glob(
        str(_ROOT / "frontend" / "checkpoints" / "**" / "best_model.pt"), recursive=True)],
    _BACKEND / "checkpoints" / "best_model.pt",
    *[Path(p) for p in glob.glob(str(_ROOT / "**" / "best_model.pt"), recursive=True)],
]

def _first_existing(paths):
    seen = set()
    for p in paths:
        p = Path(p)
        if p not in seen and p.exists():
            return p
        seen.add(p)
    return None

# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────
IMG_SIZE = 224
MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_numpy(face_rgb):
    img = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis, :]

def preprocess_torch(face_rgb):
    import torch
    return torch.from_numpy(preprocess_numpy(face_rgb)).float()

# ─────────────────────────────────────────────
#  Face Detector
# ─────────────────────────────────────────────
def get_face_detector():
    det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return det

def extract_face(frame_bgr, detector):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    fh, fw = frame_bgr.shape[:2]
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(w * 0.2); pad_y = int(h * 0.2)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
        face = frame_bgr[y1:y2, x1:x2]
        return (cv2.resize(face, (IMG_SIZE, IMG_SIZE)) if face.size else frame_bgr), True
    m = int(min(fh, fw) * 0.1)
    cropped = frame_bgr[m:fh - m, m:fw - m]
    return cv2.resize(cropped if cropped.size else frame_bgr, (IMG_SIZE, IMG_SIZE)), False

# ─────────────────────────────────────────────
#  Model Manager
# ─────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.session     = None
        self.torch_model = None
        self.mode        = None
        self.loaded      = False
        self.model_path  = None
        self.val_auc     = None
        self.face_det    = get_face_detector()

    def load_onnx(self, path):
        import onnxruntime as ort
        try:
            self.session = ort.InferenceSession(
                str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.mode = "onnx"; self.loaded = True; self.model_path = str(path)
        print(f"  ✅ ONNX: {path}")

    def load_torch(self, path):
        import torch, sys
        from collections import OrderedDict
        for sp in [str(_HERE), str(_BACKEND)]:
            if sp not in sys.path: sys.path.insert(0, sp)
        try:
            from model import DeepShieldModel
            model = DeepShieldModel(pretrained=False, freeze_backbone=False)
        except Exception as e:
            print(f"  model.py unavailable ({e}) — using timm EfficientNet-B4")
            import timm, torch.nn as nn
            backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
            model    = nn.Sequential(backbone, nn.Linear(backbone.num_features, 1))
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]; self.val_auc = ckpt.get("val_auc")
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
            else:
                sd = next((v for k, v in ckpt.items() if isinstance(v, dict)), ckpt)
        elif hasattr(ckpt, "parameters"):
            ckpt.eval(); self.torch_model = ckpt
            self.mode = "torch"; self.loaded = True; self.model_path = str(path)
            return
        else:
            raise ValueError(f"Unknown checkpoint: {type(ckpt)}")
        for prefix in ("module.", "model.", "backbone.", "net."):
            if any(k.startswith(prefix) for k in sd.keys()):
                sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
                break
        model.load_state_dict(sd, strict=False)
        model.eval(); self.torch_model = model
        self.mode = "torch"; self.loaded = True; self.model_path = str(path)
        print(f"  ✅ PyTorch: {path}  (AUC: {self.val_auc})")

    def predict_frame(self, frame_bgr):
        face, face_detected = extract_face(frame_bgr, self.face_det)
        if not self.loaded:
            prob = float(np.clip(random.gauss(0.5, 0.2), 0.01, 0.99))
            return self._format(prob, face_detected, demo_mode=True)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if self.mode == "onnx":
            tensor   = preprocess_numpy(face_rgb)
            out      = np.array(self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]).squeeze()
            fake_prob = float(1 / (1 + np.exp(-float(out)))) if out.shape in ((), (1,)) else \
                        float(np.exp(out[1]) / (np.exp(out[0]) + np.exp(out[1])))
        else:
            import torch
            tensor = preprocess_torch(face_rgb)
            with torch.no_grad():
                out = self.torch_model(tensor).squeeze()
                fake_prob = float(torch.sigmoid(out).item()) if out.dim() in (0,) or out.shape[0] == 1 \
                            else float(torch.softmax(out, 0)[1].item())
        return self._format(fake_prob, face_detected)

    def _format(self, fake_prob, face_detected, demo_mode=False):
        fake_prob  = float(np.clip(fake_prob, 0.01, 0.99))
        delta      = abs(fake_prob - 0.5)
        confidence = "HIGH" if delta >= 0.30 else ("MEDIUM" if delta >= 0.10 else "LOW")
        label      = "FAKE" if fake_prob > 0.5 else "REAL"
        return {
            "fake_prob":          round(fake_prob, 4),
            "real_prob":          round(1.0 - fake_prob, 4),
            "smoothed_fake_prob": round(fake_prob, 4),
            "smoothed_label":     label,
            "label":              label,
            "confidence":         confidence,
            "face_detected":      face_detected,
            "demo_mode":          demo_mode,
            "model_loaded":       self.loaded,
            "model_type":         self.mode or "demo",
            "deepfake_score":     round(fake_prob, 4),
            "status":             "Fake" if label == "FAKE" else "Real",
            "timestamp":          datetime.utcnow().isoformat(),
        }

# ─────────────────────────────────────────────
#  Crypto (optional)
# ─────────────────────────────────────────────
class CryptoVerifier:
    def __init__(self):
        self.sessions = {}

    def generate_keypair(self):
        if not _CRYPTO_OK: return None, None
        pk  = rsa.generate_private_key(65537, 2048, default_backend())
        pub = pk.public_key()
        return (
            pk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode(),
            pub.public_bytes(serialization.Encoding.PEM,
                             serialization.PublicFormat.SubjectPublicKeyInfo).decode(),
        )

    def generate_challenge(self, user_id):
        ch = base64.b64encode(os.urandom(32)).decode()
        self.sessions[user_id] = {"challenge": ch}
        return ch

    def verify_signature(self, user_id, sig_b64, pub_pem):
        if not _CRYPTO_OK: return False
        try:
            sess = self.sessions.get(user_id)
            if not sess: return False
            pub_key = serialization.load_pem_public_key(pub_pem.encode(), default_backend())
            pub_key.verify(base64.b64decode(sig_b64), sess["challenge"].encode(),
                           padding.PKCS1v15(), hashes.SHA256())
            del self.sessions[user_id]
            return True
        except Exception:
            return False

# ─────────────────────────────────────────────
#  MediaPipe liveness helper (legacy — still used
#  by /ws/stream for per-frame landmark data,
#  but NOT for challenge pass/fail decisions)
# ─────────────────────────────────────────────
class LandmarkHelper:
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    def __init__(self):
        self.face_mesh  = None
        self._available = False
        if not _MP_AVAILABLE: return
        try:
            _sol = getattr(mp, "solutions", None)
            if _sol:
                self.face_mesh = _sol.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            else:
                import mediapipe.python.solutions.face_mesh as _fm
                self.face_mesh = _fm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._available = True
            print("  ✅ MediaPipe landmark helper ready")
        except Exception as e:
            print(f"  ⚠  MediaPipe init failed ({e})")

    def analyse(self, frame_bgr):
        fallback = {"face_detected": False, "ear": 0.0, "head_direction": "unknown"}
        if not self._available or self.face_mesh is None:
            return fallback
        try:
            h, w = frame_bgr.shape[:2]
            res  = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                return fallback
            lm = res.multi_face_landmarks[0].landmark
            def ear(idx):
                pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx]
                A = np.linalg.norm(np.array(pts[1]) - pts[5])
                B = np.linalg.norm(np.array(pts[2]) - pts[4])
                C = np.linalg.norm(np.array(pts[0]) - pts[3])
                return float((A+B)/(2*C+1e-6))
            avg_ear = (ear(self.LEFT_EYE) + ear(self.RIGHT_EYE)) / 2
            nose    = lm[1]; lc = lm[234]; rc = lm[454]
            fw_face = abs(rc.x - lc.x)
            offset  = ((nose.x - lc.x) / fw_face) if fw_face > 0 else 0.5
            direction = "center"
            if offset < 0.4:   direction = "left"
            elif offset > 0.6: direction = "right"
            return {"face_detected": True, "ear": round(avg_ear, 3), "head_direction": direction}
        except Exception:
            return fallback

# ─────────────────────────────────────────────
#  Singletons
# ─────────────────────────────────────────────
manager  = ModelManager()
crypto   = CryptoVerifier()
landmark = LandmarkHelper()

# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────
app = FastAPI(
    title="DeepShield / TruCast API",
    description="Real-time deepfake detection + multi-frame liveness verification",
    version="2.3.0",
)
app.include_router(liveness_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Mount the fixed liveness router ──
app.include_router(liveness_router)

# ─────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    sep = "=" * 58
    print(f"\n{sep}\n  DeepShield / TruCast Backend v2.3\n{sep}")
    onnx_path = _first_existing(ONNX_CANDIDATES)
    if onnx_path:
        try: manager.load_onnx(str(onnx_path)); print(sep); return
        except Exception as e: print(f"  ⚠ ONNX failed: {e}")
    pt_path = _first_existing(PT_CANDIDATES)
    if pt_path:
        try: manager.load_torch(str(pt_path)); print(sep); return
        except Exception as e: print(f"  ⚠ PyTorch failed: {e}")
    print("  ⚠  No model — DEMO mode")
    print(f"  Place deepshield.onnx → {_BACKEND / 'models'}")
    print(sep)

# ─────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status":       "running",
        "version":      "2.3.0",
        "model_loaded": manager.loaded,
        "model_type":   manager.mode or "demo",
        "model_path":   manager.model_path,
        "val_auc":      manager.val_auc,
        "demo_mode":    not manager.loaded,
        "liveness":     "multi-frame /liveness/verify (FIXED)",
    }

@app.get("/model/status")
def model_status():
    return {"loaded": manager.loaded, "mode": manager.mode,
            "model_path": manager.model_path, "val_auc": manager.val_auc}

# ─────────────────────────────────────────────
#  Demo control
# ─────────────────────────────────────────────
@app.post("/demo/{scenario}")
def set_scenario(scenario: str):
    if scenario not in ("real", "fake", "uncertain", "auto"):
        raise HTTPException(400, "Use: real / fake / uncertain / auto")
    return {"scenario": scenario, "status": "ok"}

# ─────────────────────────────────────────────
#  Deepfake Detection
# ─────────────────────────────────────────────
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    frame    = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400, "Could not decode image")
    return JSONResponse(manager.predict_frame(frame))

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    tmp = Path(f"/tmp/trucast_{file.filename}")
    tmp.write_bytes(await file.read())
    cap, results, idx = cv2.VideoCapture(str(tmp)), [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % 5 == 0:
            r = manager.predict_frame(frame); r["frame"] = idx; results.append(r)
        idx += 1
    cap.release(); tmp.unlink(missing_ok=True)
    if not results: raise HTTPException(400, "No frames processed")
    avg  = float(np.mean([r["fake_prob"] for r in results]))
    conf = "HIGH" if abs(avg-0.5) >= 0.30 else ("MEDIUM" if abs(avg-0.5) >= 0.10 else "LOW")
    return JSONResponse({
        "label": "FAKE" if avg > 0.5 else "REAL",
        "avg_fake_prob": round(avg, 4), "avg_real_prob": round(1-avg, 4),
        "fake_prob": round(avg, 4), "deepfake_score": round(avg, 4),
        "confidence": conf, "frames_analyzed": len(results),
        "frame_breakdown": results[:10], "status": "Fake" if avg>0.5 else "Real",
        "timestamp": datetime.utcnow().isoformat(),
    })

# ─────────────────────────────────────────────
#  Crypto (if cryptography installed)
# ─────────────────────────────────────────────
@app.post("/crypto/register")
def register():
    if not _CRYPTO_OK: raise HTTPException(501, "cryptography package not installed")
    priv, pub = crypto.generate_keypair()
    return {"private_key": priv, "public_key": pub}

@app.get("/crypto/challenge/{user_id}")
def get_challenge(user_id: str):
    return {"user_id": user_id, "challenge": crypto.generate_challenge(user_id)}

class VerifyReq(BaseModel):
    user_id: str; signature: str; public_key: str

@app.post("/crypto/verify")
def verify_identity(req: VerifyReq):
    ok = crypto.verify_signature(req.user_id, req.signature, req.public_key)
    return {"verified": ok, "status": "✅ Confirmed" if ok else "❌ Failed",
            "timestamp": datetime.utcnow().isoformat()}

# ─────────────────────────────────────────────
#  WebSocket — per-frame stream
#  NOTE: liveness landmarks here are informational only.
#        Pass/fail decisions come from /liveness/verify.
# ─────────────────────────────────────────────
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    recent = []
    try:
        while True:
            try:
                data  = await websocket.receive_text()
                if "," in data: data = data.split(",", 1)[1]
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_text(json.dumps({"error": "Invalid frame"})); continue
                result = manager.predict_frame(frame)
                # Attach landmark data (informational — NOT used for liveness pass/fail)
                result["liveness_landmarks"] = landmark.analyse(frame)
                recent.append(result["fake_prob"])
                if len(recent) > 5: recent.pop(0)
                result["smoothed_fake_prob"] = round(float(np.mean(recent)), 4)
                result["smoothed_label"]     = "FAKE" if result["smoothed_fake_prob"] > 0.5 else "REAL"
                await websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass

# ─────────────────────────────────────────────
#  AI Copilot
# ─────────────────────────────────────────────
class CopilotReq(BaseModel):
    question: str; context: dict = {}

@app.post("/copilot/ask")
def copilot(req: CopilotReq):
    q   = req.question.lower()
    ctx = req.context
    fp  = ctx.get("fake_prob")
    pct = round(fp*100) if fp is not None else "N/A"
    auc = f"{manager.val_auc:.4f}" if manager.val_auc else "N/A"
    mode = manager.mode or "demo"
    if any(w in q for w in ["why","flagged","fake","detected"]):
        return {"answer": f"Detected {pct}% synthetic probability using {mode} (EfficientNet-B4). Val AUC: {auc}."}
    elif any(w in q for w in ["liveness","blink","challenge","turn"]):
        return {"answer": "Liveness uses multi-frame analysis: head turns are detected by tracking face-centre X across 40 frames; blinks by eye-region brightness dips; nods by face-centre Y range. Pass/fail is decided by actual pixel/landmark changes — NOT by deepfake probability."}
    elif any(w in q for w in ["probability","percent","score"]):
        return {"answer": f"{pct}% = model is {pct}% confident this is AI-generated. >50% = FAKE. Val AUC: {auc}."}
    else:
        return {"answer": f"DeepShield Copilot ready. Model: {mode}, AUC: {auc}. Try: 'Why was this flagged?' or 'How does liveness work?'"}

# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)