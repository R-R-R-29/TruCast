from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary dummy model
def analyze_image_with_model(image):
    return 0.2  # fake probability for now

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    probability = analyze_image_with_model(image)

    return {
        "deepfake_score": probability,
        "status": "Fake" if probability > 0.5 else "Real"
    }