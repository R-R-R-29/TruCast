"""
TruCast — Liveness Verifier
POST /liveness/verify
Receives 30-40 JPEG frames, analyses ACTUAL movement, returns pass/fail.
"""
import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/liveness", tags=["liveness"])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Thresholds (lower = easier to pass) ──
CFG = {
    "turn_x_min":   0.05,   # face must shift 5% of frame width
    "nod_y_min":    0.04,   # face must span 4% of frame height vertically
    "blink_drop":   0.03,   # eye region must darken 3% below baseline
    "blink_frames": 2,      # for at least 2 consecutive frames
    "motion_min":   3.0,    # fallback: mean pixel diff
    "min_frames":   5,
    "face_ratio":   0.35,   # 35% of frames must have a face
}


def decode(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def gray_small(f):
    return cv2.cvtColor(cv2.resize(f, (320, 240)), cv2.COLOR_BGR2GRAY)


def find_face(g):
    faces = face_cascade.detectMultiScale(g, 1.1, 4, minSize=(40, 40))
    return max(faces, key=lambda f: f[2]*f[3]) if len(faces) > 0 else None


def face_centre(g):
    roi = find_face(g)
    if roi is None:
        return None
    x, y, w, h = roi
    fh, fw = g.shape
    return (x + w/2) / fw, (y + h/2) / fh


def eye_bright(g):
    roi = find_face(g)
    if roi is None:
        return None
    x, y, w, h = roi
    band = g[y: y + max(1, h//3), x: x+w]
    return float(np.mean(band)) if band.size > 0 else None


def mad(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


# ── Challenge analysers ──

def check_turn(grays, direction):
    cx_vals = [face_centre(g) for g in grays]
    valid   = [c[0] for c in cx_vals if c is not None]
    if len(valid) < 4:
        return {"passed": False, "reason": f"Face only found in {len(valid)} frames — stay centred and well-lit.", "valid_frames": len(valid)}
    n     = len(valid)
    early = float(np.mean(valid[:max(1, n//3)]))
    late  = float(np.mean(valid[-max(1, n//3):]))
    delta = late - early
    thr   = CFG["turn_x_min"]
    passed = (delta < -thr) if direction == "left" else (delta > thr)
    return {
        "passed": passed,
        "reason": f"Head {'turned' if passed else 'not turned'} {direction}. Face X moved {delta*100:+.1f}% (need {'<-' if direction=='left' else '>'}{thr*100:.0f}%).",
        "delta_x": round(delta, 4), "threshold": thr,
        "early_cx": round(early, 4), "late_cx": round(late, 4),
        "valid_frames": len(valid)
    }


def check_nod(grays):
    cy_vals = [face_centre(g) for g in grays]
    valid   = [c[1] for c in cy_vals if c is not None]
    if len(valid) < 4:
        return {"passed": False, "reason": f"Face only found in {len(valid)} frames.", "valid_frames": len(valid)}
    cy_range = float(max(valid) - min(valid))
    thr      = CFG["nod_y_min"]
    passed   = cy_range >= thr
    return {
        "passed": passed,
        "reason": f"Nod {'detected' if passed else 'not detected'}. Vertical range {cy_range*100:.1f}% (need ≥{thr*100:.0f}%).",
        "cy_range": round(cy_range, 4), "threshold": thr, "valid_frames": len(valid)
    }


def check_blink(grays):
    brights = [eye_bright(g) for g in grays]
    valid   = [b for b in brights if b is not None]
    if len(valid) < 5:
        return {"passed": False, "reason": f"Face only in {len(valid)} frames.", "valid_frames": len(valid)}
    baseline  = float(np.percentile(valid, 75))
    threshold = baseline * (1 - CFG["blink_drop"])
    streak = max_streak = 0
    for b in valid:
        if b < threshold:
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0
    passed = max_streak >= CFG["blink_frames"]
    return {
        "passed": passed,
        "reason": f"Blink {'detected' if passed else 'not detected'}. Longest closed-eye streak: {max_streak} frames (need ≥{CFG['blink_frames']}).",
        "max_streak": max_streak, "required_streak": CFG["blink_frames"],
        "baseline": round(baseline, 1), "valid_frames": len(valid)
    }


def check_motion(grays):
    if len(grays) < 2:
        return {"passed": False, "reason": "Need ≥2 frames.", "avg_mad": 0}
    diffs   = [mad(grays[i], grays[i+1]) for i in range(len(grays)-1)]
    avg_mad = float(np.mean(diffs))
    passed  = avg_mad >= CFG["motion_min"]
    return {
        "passed": passed,
        "reason": f"Motion {'detected' if passed else 'insufficient'}: {avg_mad:.2f}px (need ≥{CFG['motion_min']}).",
        "avg_mad": round(avg_mad, 3), "threshold": CFG["motion_min"]
    }


def analyse(frames_bgr, challenge):
    n = len(frames_bgr)
    if n < CFG["min_frames"]:
        return {"passed": False, "challenge": challenge,
                "message": f"Only {n} frames — need ≥{CFG['min_frames']}. Is your camera sending frames?",
                "details": {}}

    grays       = [gray_small(f) for f in frames_bgr]
    faces_found = sum(1 for g in grays if find_face(g) is not None)
    face_ok     = (faces_found / n) >= CFG["face_ratio"]

    ch = challenge.lower()
    if ch == "turn_left":
        detail = check_turn(grays, "left")
    elif ch == "turn_right":
        detail = check_turn(grays, "right")
    elif ch == "nod":
        detail = check_nod(grays)
    elif ch in ("blink", "blink_twice"):
        detail = check_blink(grays)
    else:
        detail = check_motion(grays)

    if not face_ok:
        detail["passed"] = False
        detail["reason"] = (f"Face visible in only {faces_found}/{n} frames "
                            f"— move closer, improve lighting, face the camera directly.")

    passed  = detail["passed"]
    message = ("✓ " if passed else "✗ ") + detail["reason"]
    return {"passed": passed, "challenge": challenge, "message": message,
            "frames_total": n, "faces_found": faces_found, "details": detail}


@router.post("/verify")
async def verify_liveness(
    frames:    List[UploadFile] = File(...),
    challenge: str              = Form("default"),
):
    valid = {"turn_left", "turn_right", "blink", "blink_twice", "nod", "default"}
    if challenge not in valid:
        challenge = "default"

    frames_bgr = []
    for uf in frames:
        img = decode(await uf.read())
        if img is not None:
            frames_bgr.append(img)

    if len(frames_bgr) < 2:
        raise HTTPException(400, "Need at least 2 valid JPEG frames.")

    return JSONResponse(analyse(frames_bgr, challenge))