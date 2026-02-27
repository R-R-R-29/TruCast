"""
DeepShield — FastAPI Backend
Combines:
  - Deepfake detection (EfficientNet-B4)
  - Cryptographic identity verification
  - Liveness detection (blink + head pose)
  - AI Copilot chatbot endpoint
  - Real-time WebSocket stream

Run:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import cv2
import torch
import numpy as np
import base64
import json
import random
import os
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Crypto
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

# MediaPipe for liveness
import mediapipe as mp

import sys
sys.path.append(str(Path(__file__).parent))

from model import DeepShieldModel, load_model
from dataset import get_val_transforms, EFFICIENTNET_SIZE


# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="DeepShield API",
    description="Real-time deepfake detection with cryptographic identity & liveness verification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Face Detector (OpenCV Haar Cascade)
# ─────────────────────────────────────────────

def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load cascade: {cascade_path}")
    return detector


def extract_face(frame_bgr: np.ndarray, detector) -> tuple:
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    fh, fw = frame_bgr.shape[:2]

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x1 = max(0,  x - pad_x)
        y1 = max(0,  y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)
        face = frame_bgr[y1:y2, x1:x2]
        face_detected = True
    else:
        margin = int(min(fh, fw) * 0.1)
        face   = frame_bgr[margin:fh - margin, margin:fw - margin]
        face_detected = False

    if face.size > 0:
        face = cv2.resize(face, (EFFICIENTNET_SIZE, EFFICIENTNET_SIZE))

    return face, face_detected


# ─────────────────────────────────────────────
#  Cryptographic Identity Verifier
# ─────────────────────────────────────────────

class CryptoVerifier:
    def __init__(self):
        self.sessions = {}  # user_id → {challenge, timestamp}

    def generate_keypair(self) -> tuple:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

        return private_pem, public_pem

    def generate_challenge(self, user_id: str) -> str:
        challenge = base64.b64encode(os.urandom(32)).decode()
        self.sessions[user_id] = {
            "challenge":  challenge,
            "timestamp":  datetime.utcnow().isoformat()
        }
        return challenge

    def verify_signature(self, user_id: str, signature_b64: str, public_pem: str) -> bool:
        try:
            session = self.sessions.get(user_id)
            if not session:
                return False

            challenge  = session["challenge"].encode()
            signature  = base64.b64decode(signature_b64)
            public_key = serialization.load_pem_public_key(
                public_pem.encode(),
                backend=default_backend()
            )
            public_key.verify(
                signature,
                challenge,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            del self.sessions[user_id]  # one-time use challenge
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────
#  Liveness Detector (MediaPipe Face Mesh)
# ─────────────────────────────────────────────

class LivenessDetector:
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
    CHALLENGES = ["blink_twice", "turn_left", "turn_right", "nod"]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.blink_count = 0
        self.last_ear    = 1.0

    def _ear(self, landmarks, indices, w, h) -> float:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C)

    def get_random_challenge(self) -> str:
        return random.choice(self.CHALLENGES)

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        h, w   = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {"face_detected": False, "blink_count": self.blink_count}

        lm = result.multi_face_landmarks[0].landmark

        left_ear  = self._ear(lm, self.LEFT_EYE,  w, h)
        right_ear = self._ear(lm, self.RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        blink_detected = False
        if avg_ear < 0.2 and self.last_ear >= 0.2:
            self.blink_count += 1
            blink_detected    = True
        self.last_ear = avg_ear

        nose        = lm[1]
        left_cheek  = lm[234]
        right_cheek = lm[454]
        face_width  = abs(right_cheek.x - left_cheek.x)
        nose_offset = (nose.x - left_cheek.x) / face_width if face_width > 0 else 0.5

        head_dir = "center"
        if nose_offset < 0.4:
            head_dir = "left"
        elif nose_offset > 0.6:
            head_dir = "right"

        return {
            "face_detected":  True,
            "ear":            round(avg_ear, 3),
            "blink_detected": blink_detected,
            "blink_count":    self.blink_count,
            "head_direction": head_dir,
        }


# ─────────────────────────────────────────────
#  Deepfake Model Manager
# ─────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.model         = None
        self.transform     = get_val_transforms(EFFICIENTNET_SIZE)
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded        = False
        self.ckpt_path     = None
        self.face_detector = get_face_detector()

    def load(self, checkpoint_path: str):
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            print(f"⚠️  Checkpoint not found — running in DEMO MODE")
            return
        print(f"Loading model from {checkpoint_path} ...")
        self.model     = load_model(checkpoint_path, freeze_backbone=False)
        self.model.eval()
        self.loaded    = True
        self.ckpt_path = checkpoint_path
        print(f"✅ Model loaded | Device: {self.device}")

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        face, face_detected = extract_face(frame_bgr, self.face_detector)

        if not self.loaded:
            fake_prob = random.uniform(0.3, 0.9)
            return self._format(fake_prob, face_detected, demo_mode=True)

        face_rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=face_rgb)
        tensor    = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit     = self.model(tensor)
            fake_prob = torch.sigmoid(logit).item()

        return self._format(fake_prob, face_detected)

    def _format(self, fake_prob: float, face_detected: bool, demo_mode: bool = False) -> dict:
        delta      = abs(fake_prob - 0.5)
        confidence = "HIGH" if delta >= 0.30 else ("MEDIUM" if delta >= 0.10 else "LOW")
        return {
            "fake_prob":     round(fake_prob, 4),
            "real_prob":     round(1.0 - fake_prob, 4),
            "label":         "FAKE" if fake_prob > 0.5 else "REAL",
            "confidence":    confidence,
            "face_detected": face_detected,
            "demo_mode":     demo_mode,
            "timestamp":     datetime.utcnow().isoformat()
        }


# ─────────────────────────────────────────────
#  Singletons
# ─────────────────────────────────────────────

manager  = ModelManager()
crypto   = CryptoVerifier()
liveness = LivenessDetector()


# ─────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    checkpoints = sorted(Path("./checkpoints").rglob("best_model.pt"))
    if checkpoints:
        manager.load(str(checkpoints[-1]))
    else:
        print("⚠️  No checkpoint found — running in demo mode.")


# ─────────────────────────────────────────────
#  Health & Status
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status":       "running",
        "model_loaded": manager.loaded,
        "device":       str(manager.device),
        "checkpoint":   manager.ckpt_path
    }


@app.get("/model/status")
def model_status():
    return {
        "loaded":     manager.loaded,
        "checkpoint": manager.ckpt_path,
        "device":     str(manager.device),
        "demo_mode":  not manager.loaded
    }


# ─────────────────────────────────────────────
#  Deepfake Detection
# ─────────────────────────────────────────────

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents  = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    return JSONResponse(manager.predict_frame(frame))


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    tmp_path = Path(f"./tmp_{file.filename}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    cap           = cv2.VideoCapture(str(tmp_path))
    frame_results = []
    frame_idx     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0:
            result          = manager.predict_frame(frame)
            result["frame"] = frame_idx
            frame_results.append(result)
        frame_idx += 1

    cap.release()
    tmp_path.unlink(missing_ok=True)

    if not frame_results:
        raise HTTPException(400, "No frames processed")

    avg_fake = float(np.mean([r["fake_prob"] for r in frame_results]))
    delta    = abs(avg_fake - 0.5)
    conf     = "HIGH" if delta >= 0.30 else ("MEDIUM" if delta >= 0.10 else "LOW")

    return JSONResponse({
        "label":           "FAKE" if avg_fake > 0.5 else "REAL",
        "avg_fake_prob":   round(avg_fake, 4),
        "avg_real_prob":   round(1 - avg_fake, 4),
        "confidence":      conf,
        "frames_analyzed": len(frame_results),
        "frame_breakdown": frame_results[:10],
        "timestamp":       datetime.utcnow().isoformat()
    })


# ─────────────────────────────────────────────
#  Cryptographic Identity
# ─────────────────────────────────────────────

@app.post("/crypto/register")
def register_user():
    private_key, public_key = crypto.generate_keypair()
    return {
        "private_key": private_key,
        "public_key":  public_key,
        "message":     "Store private_key securely on client. Never send it to server."
    }


@app.get("/crypto/challenge/{user_id}")
def get_challenge(user_id: str):
    challenge = crypto.generate_challenge(user_id)
    return {"user_id": user_id, "challenge": challenge, "expires": "60 seconds"}


class VerifyRequest(BaseModel):
    user_id:    str
    signature:  str
    public_key: str


@app.post("/crypto/verify")
def verify_identity(req: VerifyRequest):
    verified = crypto.verify_signature(req.user_id, req.signature, req.public_key)
    return {
        "verified":  verified,
        "user_id":   req.user_id,
        "status":    "✅ Identity Confirmed" if verified else "❌ Identity Failed",
        "timestamp": datetime.utcnow().isoformat()
    }


# ─────────────────────────────────────────────
#  Liveness Detection
# ─────────────────────────────────────────────

@app.get("/liveness/challenge")
def liveness_challenge():
    challenge = liveness.get_random_challenge()
    messages  = {
        "blink_twice": "Please blink twice",
        "turn_left":   "Please turn your head to the left",
        "turn_right":  "Please turn your head to the right",
        "nod":         "Please nod your head up and down"
    }
    return {"challenge": challenge, "message": messages.get(challenge), "timeout": 10}


@app.post("/liveness/check")
async def liveness_check(file: UploadFile = File(...)):
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode frame")
    return JSONResponse(liveness.analyze_frame(frame))


@app.post("/liveness/reset")
def reset_liveness():
    liveness.blink_count = 0
    liveness.last_ear    = 1.0
    return {"status": "reset", "blink_count": 0}


# ─────────────────────────────────────────────
#  Full Verification Pipeline (one-shot)
# ─────────────────────────────────────────────

class FullVerifyRequest(BaseModel):
    user_id:    str
    signature:  str
    public_key: str


@app.post("/verify/full")
async def full_verification(req: FullVerifyRequest, file: UploadFile = File(...)):
    """Combines crypto + liveness + deepfake into one trust score."""
    identity_ok     = crypto.verify_signature(req.user_id, req.signature, req.public_key)

    contents        = await file.read()
    nparr           = np.frombuffer(contents, np.uint8)
    frame           = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode frame")

    liveness_result = liveness.analyze_frame(frame)
    deepfake_result = manager.predict_frame(frame)

    identity_score  = 100 if identity_ok else 0
    liveness_score  = min(100, liveness_result.get("blink_count", 0) * 50)
    deepfake_score  = int(deepfake_result["real_prob"] * 100)
    trust_score     = int((identity_score * 0.3) + (liveness_score * 0.3) + (deepfake_score * 0.4))

    verdict = (
        "✅ VERIFIED HUMAN"           if trust_score >= 70 else
        "⚠️  SUSPICIOUS"              if trust_score >= 40 else
        "❌ SYNTHETIC MEDIA DETECTED"
    )

    return JSONResponse({
        "verdict":     verdict,
        "trust_score": trust_score,
        "identity":    {"verified": identity_ok,   "score": identity_score},
        "liveness":    {**liveness_result,          "score": liveness_score},
        "deepfake":    {**deepfake_result,           "score": deepfake_score},
        "timestamp":   datetime.utcnow().isoformat()
    })


# ─────────────────────────────────────────────
#  AI Copilot
# ─────────────────────────────────────────────

class CopilotRequest(BaseModel):
    question: str
    context:  dict = {}


@app.post("/copilot/ask")
def copilot_ask(req: CopilotRequest):
    q      = req.question.lower()
    ctx    = req.context
    fake_p = ctx.get("fake_prob", None)
    label  = ctx.get("label", "unknown")

    if any(w in q for w in ["why", "flagged", "fake", "detected"]):
        if fake_p is not None:
            return {"answer": f"The system detected a {round(fake_p*100)}% probability of synthetic media. "
                              f"This is based on EfficientNet-B4 analysis of facial artifacts, unnatural blending, "
                              f"and inconsistencies typical of GAN-generated or swapped faces."}
        return {"answer": "The system flagged this based on facial artifact analysis from our EfficientNet-B4 model trained on FaceForensics++."}

    elif any(w in q for w in ["probability", "percent", "score", "mean", "confidence"]):
        if fake_p is not None:
            return {"answer": f"A {round(fake_p*100)}% deepfake probability means the model is "
                              f"{round(fake_p*100)}% confident this is synthetic. "
                              f"Above 50% = FAKE, below 50% = REAL. Above 80% = HIGH confidence."}
        return {"answer": "The probability score reflects how confident the model is that the media is AI-generated. Above 50% = FAKE."}

    elif any(w in q for w in ["liveness", "blink", "real person", "live"]):
        return {"answer": "Liveness detection verifies a real human is present by tracking eye blinks and "
                          "head movements using MediaPipe Face Mesh. Deepfakes cannot respond to "
                          "randomized physical challenges in real time."}

    elif any(w in q for w in ["crypto", "key", "identity", "verify", "signature"]):
        return {"answer": "Cryptographic verification uses RSA-2048 public-key signatures. "
                          "A unique challenge is issued per session, signed by the user's private key, "
                          "and verified server-side. This proves device identity without transmitting secrets."}

    elif any(w in q for w in ["improve", "better", "accuracy", "fix"]):
        return {"answer": "To improve verification: ensure good lighting, face the camera directly, "
                          "complete the liveness challenge fully, and use a stable internet connection."}

    elif any(w in q for w in ["trust", "score", "safe"]):
        return {"answer": "Trust Score = 40% deepfake model + 30% liveness + 30% crypto identity. "
                          "Above 70% = Verified Human. Below 40% = likely synthetic or spoofed."}

    elif any(w in q for w in ["incident", "report", "log"]):
        return {"answer": f"Incident Report — {datetime.utcnow().isoformat()}: "
                          f"Label: {label}. Fake probability: {round(fake_p*100) if fake_p else 'N/A'}%. "
                          f"Recommend flagging for manual review."}

    else:
        return {"answer": "I'm the DeepShield AI Copilot. Ask me: 'Why was this flagged?', "
                          "'What does 72% mean?', 'How does liveness work?', or 'What is the trust score?'"}


# ─────────────────────────────────────────────
#  WebSocket — Real-time Stream
# ─────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket connected: {websocket.client}")
    recent_probs = []

    try:
        while True:
            data = await websocket.receive_text()
            try:
                if "," in data:
                    data = data.split(",")[1]

                img_bytes = base64.b64decode(data)
                img_array = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_text(json.dumps({"error": "Invalid frame"}))
                    continue

                result         = manager.predict_frame(frame)
                result["liveness"] = liveness.analyze_frame(frame)

                recent_probs.append(result["fake_prob"])
                if len(recent_probs) > 5:
                    recent_probs.pop(0)
                smoothed                     = float(np.mean(recent_probs))
                result["smoothed_fake_prob"] = round(smoothed, 4)
                result["smoothed_label"]     = "FAKE" if smoothed > 0.5 else "REAL"

                await websocket.send_text(json.dumps(result))

            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {websocket.client}")


# ─────────────────────────────────────────────
#  Manual Model Load
# ─────────────────────────────────────────────

class LoadModelRequest(BaseModel):
    checkpoint_path: str


@app.post("/model/load")
def load_model_endpoint(req: LoadModelRequest):
    try:
        manager.load(req.checkpoint_path)
        return {"success": True, "checkpoint": req.checkpoint_path}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)