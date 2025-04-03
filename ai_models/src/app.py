from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
import json
from typing import Dict, Any

app = FastAPI()

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models')
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'threat_detector.h5'))
    except Exception as e:
        print(f"Error loading model: {e}")
        # For testing, create a dummy model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

class AnalysisRequest(BaseModel):
    data: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    try:
        # Convert input data to numpy array
        input_data = np.array(list(request.data.values())).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0][0]
        
        # Determine threat level
        threat_level = "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.3 else "LOW"
        
        return {
            "threat_level": threat_level,
            "confidence": float(prediction),
            "details": {
                "raw_prediction": float(prediction),
                "thresholds": {
                    "high": 0.7,
                    "medium": 0.3
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 3002))
    uvicorn.run(app, host="0.0.0.0", port=port) 