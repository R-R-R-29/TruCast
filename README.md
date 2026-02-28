## 📌 Project Description

DeepShield is a real-time AI-powered deepfake detection and liveness verification system designed to identify spoofing attacks during live video sessions.

With the rapid growth of AI-generated deepfake videos, biometric authentication systems face serious security risks. DeepShield analyzes facial movement patterns, frame consistency, and micro-expressions using a deep learning model (EfficientNet-B4) to classify input as:

* 🟢 Real Human
* 🔴 Deepfake
* 🟡 Uncertain

The system can be integrated into authentication platforms, banking apps, proctoring systems, and identity verification workflows to prevent fraud and identity misuse.

---

## 🛠️ Tech Stack

### 🌐 Frontend

* React.js
* Vite
* CSS
* HTML
* Javascript

### 🔧 Backend

* Python
* FastAPI
* Uvicorn

### 🧠 AI / Machine Learning

* TensorFlow
* EfficientNet-B4
* OpenCV
* NumPy

### ☁️ Deployment

* Backend: Render / Railway


---

## 🚀 Features

* 🎥 Real-time camera feed analysis
* 🧠 Deepfake detection using EfficientNet-B4
* ⚡ Auto detection cycle mode
* 🎯 Trigger-based liveness challenge
* 📊 Live session statistics (Frames, Passed, Failed)
* 🔍 Confidence-based prediction scoring
* 🌐 REST API for frame-based analysis
* 📈 Lightweight and modular architecture

---

## 🏗️ System Architecture

```
User Camera
     │
     ▼
Frontend (React)
     │
     ▼
Backend API (FastAPI)
     │
     ▼
AI Model (EfficientNet-B4)
     │
     ▼
Prediction Response (Real / Fake / Uncertain)
```

📂 Architecture diagram file:
`docs/architecture.png`

---

## 📸 Screenshots
https://drive.google.com/file/d/10O4mLObzt3wEhNnN9nqUV0yVFFcpKmS2/view?usp=sharing
https://drive.google.com/file/d/1vk1JUsJFdqvWDIXvLAf2LsodPWSMp6nj/view?usp=sharing
https://drive.google.com/file/d/1zNMbMq9UGNo0qqMfcETwujPR5ki7QBin/view?usp=sharing


---

## 🎬 Demo Video

Watch the demo here:
👉 https://drive.google.com/file/d/1TTGarBkXTly8-IbWEcVmTy1uioAQ3Uxh/view?usp=sharing

(Ensure link is public and under 3 minutes)

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/R-R-R-29/TruCast.git
cd TruCast
```

---

### 2️⃣ Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## ▶️ Running the Project

### Start Backend Server

```bash
uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### Start Frontend

```bash
npm run dev
```

Frontend runs at:

```
http://localhost:5173/
```

---

## 🔌 API Documentation
http://127.0.0.1:8000

┌───────────────────────────────┐
│           Frontend            │
│      (React / HTML / JS)      │
│                               │
│  • Video Capture              │
│  • Frame Extraction           │
│  • UI Alerts & Results        │
└───────────────┬───────────────┘
                │
                │ HTTP / WebSocket
                ▼
┌───────────────────────────────┐
│          Backend API          │
│      (FastAPI + Uvicorn)      │
│                               │
│  • REST Endpoints             │
│  • WebSocket Streaming        │
│  • Image Preprocessing        │
└───────────────┬───────────────┘
                │
                │ Model Inference
                ▼
┌───────────────────────────────┐
│        DeepShield Model       │
│       (EfficientNet-B4)       │
│                               │
│  • Feature Extraction         │
│  • Binary Classification       │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│       Prediction Response     │
│                               │
│  • Real / Fake                │
│  • Confidence Score           │
└───────────────────────────────┘
### 📍 POST /analyze

Analyzes an image frame and returns deepfake classification.

#### Request (JSON)

```json
{
  "image": "base64_encoded_string"
}
```

#### Response

```json
{
  "fake_probability": 0.87,
  "label": "FAKE"
}
```

---

## 📂 Project Structure

```
trucast/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── routes.py
│   │   ├── liveness.py
│   │   └── crypto_verify.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── docs/
│   ├── architecture.png
│   └── flow_diagram.png
│
├── README.md
├── LICENSE
└── .gitignore
```

✔ Lowercase folder names
✔ No spaces in filenames
✔ Organized modular structure

---

## 🌍 Live Deployment

Frontend:
👉 [https://your-live-frontend-link.com](https://your-live-frontend-link.com)

Backend API:
👉 [https://your-live-backend-link.com](https://your-live-backend-link.com)

* HTTPS enabled
* No runtime errors
* Optimized production build

---

## 🧪 Model Performance

* Accuracy: 93%
* Precision: 91%
* Recall: 92%
* F1-Score: 91%

(Replace with your real metrics if available)

---

## 🤖 AI Tools Used

* ChatGPT – Documentation structuring and architecture planning
* GitHub Copilot – Code suggestions


---

## 👥 Team Members

* Riya Rebecca Renjit 1 – AI Model Development
* Sruthi A S 2 – Frontend Development

---

## 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for more details.

---

## 🔮 Future Improvements

* Multi-face detection support
* Improved model confidence calibration
* Edge-device deployment (mobile support)
* Advanced anti-spoofing challenge-response
* Integration with authentication APIs

---


