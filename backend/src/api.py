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
from pydantic import BaseModel, Field, validator, RootModel
import numpy as np
import time
import gc
from dotenv import load_dotenv

# Add the root directory to the path to find the models module
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import after adding to path
from models.threat_detection_model import ThreatDetectionModel
from models.metrics import update_model_metrics, record_prediction, record_batch_size, update_gpu_memory, record_prediction_result, set_model_version
import tensorflow as tf
import psutil
import requests

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

# Add configuration for blockchain integration
BLOCKCHAIN_WEBHOOK_URL = os.environ.get('BLOCKCHAIN_WEBHOOK_URL', 'http://localhost:3000/api/v1/ai-detection')
BLOCKCHAIN_ENABLED = os.environ.get('BLOCKCHAIN_ENABLED', 'true').lower() == 'true'

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

class AnalysisRequest(BaseModel):
    """Request model for analyzing network traffic"""
    data: List[List[float]] = Field(..., description="List of network traffic samples")

class AnalysisResponse(BaseModel):
    """Response model for analyzing network traffic"""
    results: List[ThreatPrediction] = Field(..., description="List of threat predictions")

class ThreatPrediction(BaseModel):
    """Model for threat prediction"""
    threat_level: str = Field(..., description="Threat level")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each threat type")
    model_version: str = Field(..., description="Model version")

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

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze network traffic for threats"""
    
    try:
        # ... existing processing code ...

        # Process results
        results = []
        for i, probs in enumerate(probabilities):
            confidence = np.max(probs)
            predicted_class = np.argmax(probs)
            
            # Determine threat level based on confidence
            threat_level = "low"
            if confidence > 0.9:
                threat_level = "critical"
            elif confidence > 0.7:
                threat_level = "high"
            elif confidence > 0.4:
                threat_level = "medium"
            
            # Map class index to threat type
            threat_type = "Unknown"
            if predicted_class == 0:
                threat_type = "Normal"
            elif predicted_class == 1:
                threat_type = "DDoS"
            elif predicted_class == 2:
                threat_type = "Brute Force"
            elif predicted_class == 3:
                threat_type = "Port Scan"
            elif predicted_class == 4:
                threat_type = "Malware"
            
            # Create result object
            result = ThreatPrediction(
                threat_level=threat_level,
                confidence=float(confidence),
                probabilities={
                    "normal": float(probs[0]) if len(probs) > 0 else 0.0,
                    "ddos": float(probs[1]) if len(probs) > 1 else 0.0,
                    "brute_force": float(probs[2]) if len(probs) > 2 else 0.0,
                    "port_scan": float(probs[3]) if len(probs) > 3 else 0.0,
                    "malware": float(probs[4]) if len(probs) > 4 else 0.0
                },
                model_version=model.model_version
            )
            results.append(result)
            
            # Record prediction result for metrics
            record_prediction_result(threat_level)
            
            # Send significant threats to blockchain if enabled
            if BLOCKCHAIN_ENABLED and threat_level != "low" and threat_type != "Normal":
                try:
                    # Prepare data for blockchain
                    blockchain_data = {
                        "threat_type": threat_type,
                        "confidence": float(confidence),
                        "raw_predictions": [float(p) for p in probs],
                        "source_data": request.data[i],
                        "timestamp": time.time(),
                        "model_version": model.model_version
                    }
                    
                    # Send to blockchain webhook in the background to avoid blocking
                    background_tasks.add_task(
                        send_to_blockchain,
                        blockchain_data
                    )
                except Exception as e:
                    logging.error(f"Error sending to blockchain: {str(e)}")

        # ... rest of the existing code ...
        
    except Exception as e:
        logging.error(f"Error in analyze_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def send_to_blockchain(event_data):
    """Send threat event data to the blockchain integration service"""
    try:
        # Send data to the blockchain webhook
        response = requests.post(
            BLOCKCHAIN_WEBHOOK_URL,
            json=event_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Log the result
        if response.status_code == 202:
            logging.info(f"Event sent to blockchain service: {response.json().get('message', 'Success')}")
        else:
            logging.error(f"Failed to send event to blockchain: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error in send_to_blockchain: {str(e)}")

# Add a new endpoint to get recent threats for polling mode
@app.get("/api/recent-threats")
async def recent_threats():
    """Get recent threats for blockchain integration in polling mode"""
    # This would typically retrieve from a database or cache
    # For now, we'll return a simple implementation with mock data if needed
    try:
        # In a real implementation, you would retrieve this from a database
        # For now, return empty if no recent threats
        return {"threats": []}
    except Exception as e:
        logging.error(f"Error getting recent threats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.route('/api/blockchain/verify/<event_id>', methods=['GET'])
def verify_blockchain_event(event_id):
    """Verify a blockchain event and generate a verification certificate"""
    try:
        # In a real implementation, this would call the blockchain adapter
        # For development/testing, we can use a simple mock
        if os.environ.get('USE_MOCK_BLOCKCHAIN') == 'true':
            # Return a mock verification certificate
            return jsonify({
                "eventId": event_id,
                "type": "SECURITY_EVENT",
                "timestamp": "2023-06-15T10:30:15.123Z",
                "blockchainTimestamp": "2023-06-15T10:30:45.789Z",
                "blockNumber": "12345",
                "transactionId": "abcdef1234567890abcdef1234567890",
                "dataHash": "8a1bc7a5d0e96c273d974b3b713f11067193ebca4efe7266e249d7774ebcc67d",
                "verificationTimestamp": datetime.now().isoformat(),
                "status": "VERIFIED",
                "verificationDetails": {
                    "dataIntegrity": {
                        "isValid": True,
                        "calculatedHash": "8a1bc7a5d0e96c273d974b3b713f11067193ebca4efe7266e249d7774ebcc67d",
                        "storedHash": "8a1bc7a5d0e96c273d974b3b713f11067193ebca4efe7266e249d7774ebcc67d"
                    },
                    "transactionValidity": {
                        "isValid": True,
                        "txId": "abcdef1234567890abcdef1234567890",
                        "blockNumber": "12345",
                        "timestamp": "2023-06-15T10:30:45.789Z",
                        "validationCode": 0
                    }
                }
            })
        else:
            # In production, we would call the actual blockchain adapter
            # This requires integrating with the Node.js blockchain-adapter.js
            from subprocess import Popen, PIPE
            import json
            
            # Call the Node.js verification script
            process = Popen(
                ['node', '../scripts/verify-blockchain-event.js', event_id],
                stdout=PIPE,
                stderr=PIPE
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Verification failed: {stderr.decode('utf-8')}")
            
            # Parse the result
            result = json.loads(stdout.decode('utf-8'))
            return jsonify(result)
    except Exception as e:
        app.logger.error(f"Blockchain verification error: {str(e)}")
        return jsonify({"error": str(e)}), 500

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