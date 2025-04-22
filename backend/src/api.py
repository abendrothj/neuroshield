#!/usr/bin/env python3

"""
NeuraShield API Service
This provides a RESTful API for the NeuraShield threat detection system using FastAPI.
"""

import os
import sys
import logging
import json
import uvicorn
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the predictor class from inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import NeuraShieldPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/api.log"))
    ]
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)

# Initialize the predictor
predictor = NeuraShieldPredictor()

# Create the FastAPI application
app = FastAPI(
    title="NeuraShield API",
    description="API for NeuraShield cybersecurity threat detection",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define Pydantic models for request and response
class NetworkTrafficFeatures(BaseModel):
    """Network traffic features for prediction"""
    feature_0: float = Field(..., description="Feature 0 (e.g., flow duration)")
    feature_1: float = Field(..., description="Feature 1 (e.g., bytes transferred)")
    feature_2: float = Field(..., description="Feature 2 (e.g., packet count)")
    feature_3: float = Field(..., description="Feature 3 (e.g., flow rate)")
    feature_4: float = Field(..., description="Feature 4 (e.g., TTL value)")
    feature_5: float = Field(..., description="Feature 5 (e.g., window size)")
    feature_6: float = Field(..., description="Feature 6 (e.g., payload size)")
    feature_7: float = Field(..., description="Feature 7 (e.g., inter-arrival time)")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_0": 0.25,
                "feature_1": 0.75,
                "feature_2": 0.1,
                "feature_3": 0.5,
                "feature_4": 0.3,
                "feature_5": 0.01,
                "feature_6": 0.8,
                "feature_7": 0.05
            }
        }

class TrafficBatch(BaseModel):
    """Batch of network traffic samples for prediction"""
    samples: List[NetworkTrafficFeatures] = Field(..., description="List of network traffic samples")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: str = Field(..., description="Prediction result (attack or benign)")
    probability: float = Field(..., description="Probability of the prediction")
    confidence: float = Field(..., description="Confidence score for the prediction")
    threshold: float = Field(..., description="Classification threshold used")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint"""
    results: List[PredictionResponse] = Field(..., description="List of prediction results")
    summary: Dict[str, int] = Field(..., description="Summary of predictions (count of each class)")

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")

# Background task for logging predictions
def log_prediction(prediction: Dict[str, Any], traffic_data: Dict[str, Any]):
    """Log prediction to file"""
    log_entry = {
        "timestamp": prediction.get("timestamp"),
        "prediction": prediction.get("prediction"),
        "probability": prediction.get("probability"),
        "traffic_data": traffic_data
    }
    
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/predictions.jsonl")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# API routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "NeuraShield API",
        "description": "API for network threat detection powered by multi-dataset transfer learning",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is operational"""
    return {
        "status": "healthy",
        "model_loaded": predictor.loaded,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(traffic: NetworkTrafficFeatures, background_tasks: BackgroundTasks):
    """
    Make a prediction for a single traffic sample
    
    Args:
        traffic: Network traffic features
        
    Returns:
        Prediction result with probability and confidence
    """
    from datetime import datetime
    
    # Convert Pydantic model to dict
    traffic_dict = traffic.dict()
    
    # Make prediction
    result = predictor.predict(traffic_dict)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Add timestamp
    result["timestamp"] = datetime.now().isoformat()
    
    # Log prediction in background
    background_tasks.add_task(log_prediction, result, traffic_dict)
    
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: TrafficBatch, background_tasks: BackgroundTasks):
    """
    Make predictions for a batch of traffic samples
    
    Args:
        batch: Batch of network traffic samples
        
    Returns:
        List of prediction results and summary statistics
    """
    from datetime import datetime
    
    # Convert Pydantic models to dicts
    sample_dicts = [sample.dict() for sample in batch.samples]
    
    # Make batch prediction
    results = predictor.predict_batch(sample_dicts)
    
    if results and "error" in results[0]:
        raise HTTPException(status_code=500, detail=results[0]["error"])
    
    # Add timestamp to each result
    timestamp = datetime.now().isoformat()
    for result in results:
        result["timestamp"] = timestamp
    
    # Generate summary
    summary = {
        "total": len(results),
        "attack": sum(1 for r in results if r["prediction"] == "attack"),
        "benign": sum(1 for r in results if r["prediction"] == "benign")
    }
    
    # Log each prediction in background
    for i, result in enumerate(results):
        background_tasks.add_task(log_prediction, result, sample_dicts[i])
    
    return {
        "results": results,
        "summary": summary
    }

@app.post("/explain", response_model=Dict[str, Any])
async def explain(traffic: NetworkTrafficFeatures):
    """
    Explain a prediction by showing feature contributions
    
    Args:
        traffic: Network traffic features
        
    Returns:
        Prediction result with feature contribution information
    """
    # Convert Pydantic model to dict
    traffic_dict = traffic.dict()
    
    # Get explanation
    result = predictor.explain(traffic_dict)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions and return a JSON response"""
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )

def main():
    """Run the API server"""
    # If running as script, start the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 