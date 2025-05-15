from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os

# Import recognizers
from webcam_en import ASLRecognizer as EnglishRecognizer
from webcam_ar import ASLRecognizerArabic as ArabicRecognizer

app = FastAPI()

# Allow CORS (for local or Flutter WebView)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # ← Change this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend if needed
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# Prediction endpoint for both Arabic and English
@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if language == "en":
        recognizer = EnglishRecognizer()
        label, confidence = recognizer.process_frame(frame)
    elif language == "ar":
        recognizer = ArabicRecognizer()
        label, confidence = recognizer.process_frame(frame)
    else:
        raise HTTPException(status_code=400, detail="Unsupported language")

    return {
        "prediction": label,
        "confidence": float(confidence)
    }