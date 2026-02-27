'''
import cv2
import mediapipe as mp
import numpy as np
import random
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

app = FastAPI()

class LivenessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # Eye landmark indices
        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33,  160, 158, 133, 153, 144]

        self.challenges = [
            "blink_twice",
            "turn_left",
            "turn_right",
            "nod"
        ]
        self.blink_count    = 0
        self.last_blink_ear = 1.0

    def eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C)

    def get_random_challenge(self):
        return random.choice(self.challenges)

    def analyze_frame(self, frame):
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {"face_detected": False}

        landmarks = result.multi_face_landmarks[0].landmark

        # Blink detection
        left_ear  = self.eye_aspect_ratio(landmarks, self.LEFT_EYE,  w, h)
        right_ear = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        blink_detected = False
        if avg_ear < 0.2 and self.last_blink_ear >= 0.2:
            self.blink_count += 1
            blink_detected = True
        self.last_blink_ear = avg_ear

        # Head pose (yaw estimation)
        nose    = landmarks[1]
        left_cheek  = landmarks[234]
        right_cheek = landmarks[454]
        face_width  = abs(right_cheek.x - left_cheek.x)
        nose_offset = (nose.x - left_cheek.x) / face_width if face_width > 0 else 0.5

        head_direction = "center"
        if nose_offset < 0.4:
            head_direction = "left"
        elif nose_offset > 0.6:
            head_direction = "right"

        return {
            "face_detected":   True,
            "ear":             round(avg_ear, 3),
            "blink_detected":  blink_detected,
            "blink_count":     self.blink_count,
            "head_direction":  head_direction,
        }

liveness_detector = LivenessDetector()

'''