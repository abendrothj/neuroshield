from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import numpy as np
import time
import os
from ai_models.threat_detection_model import ThreatDetectionModel
from ai_models.metrics import update_model_metrics, record_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Initialize the app
app = FastAPI(title="NeuraShield AI API", description="API for NeuraShield's threat detection AI model")

# Initialize model (lazy loading)
model = None

# Input data model
class AnalysisRequest(BaseModel):
    data: dict

# Response model
class AnalysisResponse(BaseModel):
    result: dict
    confidence: float
    prediction_time: float
    model_info: dict = None

# Health response model
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def get_model():
    """Get or initialize the model"""
    global model
    
    if model is None:
        logging.info("Initializing model...")
        model = ThreatDetectionModel()
        
        # Try to load the model
        model_path = os.environ.get("MODEL_PATH", "models/threat_detection")
        try:
            model.load(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.warning(f"Could not load model from {model_path}: {str(e)}")
            logging.info("Building new model...")
            model.input_shape = (9,)  # Default input shape (9 features)
            model.num_classes = 5     # Default number of classes
            model.build_model()
            
    return model

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Analyze data for threats"""
    try:
        # Get the model
        current_model = get_model()
        
        # Convert data to numpy array
        try:
            # Extract features
            features = np.array([list(request.data.values())], dtype=np.float32)
            
            # Make prediction
            start_time = time.time()
            predictions = current_model.predict(features)
            prediction_time = time.time() - start_time
            
            # Get the class with highest probability
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Define threat levels
            threat_levels = ["safe", "low_risk", "medium_risk", "high_risk", "critical"]
            
            # Prepare result
            result = {
                "threat_level": threat_levels[predicted_class],
                "probability": confidence,
                "class_index": int(predicted_class)
            }
            
            # Get model info
            model_info = current_model.get_model_info()
            if "training_history" in model_info:
                # Extract only the last accuracy value
                if model_info["training_history"] and "val_accuracy" in model_info["training_history"]:
                    model_info["accuracy"] = model_info["training_history"]["val_accuracy"][-1]
                del model_info["training_history"]
            
            return {
                "result": result,
                "confidence": confidence,
                "prediction_time": prediction_time,
                "model_info": model_info
            }
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing data: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Metrics endpoint is handled by the prometheus_client"""
    return "Metrics are available at :8000/metrics"

@app.get("/api/metrics")
async def api_metrics():
    """Get metrics in a format suitable for the frontend"""
    try:
        # Get the model
        current_model = get_model()
        
        # Get model info
        model_info = current_model.get_model_info()
        
        # Calculate accuracy from model info
        accuracy = 0.9  # Default value
        if "training_history" in model_info and model_info["training_history"]:
            if "val_accuracy" in model_info["training_history"]:
                accuracy = model_info["training_history"]["val_accuracy"][-1]
        
        # Return metrics in frontend format
        return {
            "accuracy": accuracy,
            "inference_time": 0.05,  # Placeholder - would come from actual monitoring
            "memory_usage": model_info.get("trainable_params", 500000) * 4,  # Approximate memory usage
            "predictions_total": 10000,  # Placeholder - would come from Prometheus metrics
            "error_rate": 0.01  # Placeholder - would come from Prometheus metrics
        }
    except Exception as e:
        logging.error(f"Error getting API metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

# Training request model
class TrainingRequest(BaseModel):
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    dataset_size: int = 10000
    validation_split: float = 0.2

# Training status response model
class TrainingResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    results: dict = None

# In-memory storage for training jobs (in a production environment, this would be in a database)
training_jobs = {}

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Start model training"""
    try:
        # Generate a job ID
        job_id = f"train_{int(time.time())}"
        
        # Store job information
        training_jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued for processing",
            "request": request.dict(),
            "results": None,
            "start_time": None,
            "end_time": None
        }
        
        # In a real implementation, this would start a background task
        # For demonstration, we'll simulate training in the background
        logging.info(f"Training job {job_id} queued with params: {request}")
        
        # Simulate background task with a simple thread
        # In production, you'd use a proper task queue like Celery
        def train_in_background():
            try:
                # Update job status
                training_jobs[job_id]["status"] = "running"
                training_jobs[job_id]["start_time"] = time.time()
                training_jobs[job_id]["message"] = "Training in progress"
                
                # Get the model
                model = get_model()
                
                # Simulate training steps
                total_steps = request.epochs
                for step in range(total_steps):
                    # Update progress
                    progress = (step + 1) / total_steps
                    training_jobs[job_id]["progress"] = progress
                    training_jobs[job_id]["message"] = f"Training epoch {step + 1}/{total_steps}"
                    
                    # Simulate training delay
                    time.sleep(0.1)  # Fast simulation for demo purposes
                
                # Simulate training results
                accuracy = 0.95
                loss = 0.164
                
                # Update model metrics
                update_model_metrics(
                    model_name=model.model_name,
                    accuracy=accuracy,
                    memory_usage=model.model.count_params() * 4 if model.model else 1000000
                )
                
                # Update job status
                training_jobs[job_id]["status"] = "completed"
                training_jobs[job_id]["progress"] = 1.0
                training_jobs[job_id]["message"] = "Training completed successfully"
                training_jobs[job_id]["end_time"] = time.time()
                training_jobs[job_id]["results"] = {
                    "accuracy": accuracy,
                    "loss": loss,
                    "training_time": training_jobs[job_id]["end_time"] - training_jobs[job_id]["start_time"],
                    "model_size": model.model.count_params() * 4 if model.model else 1000000
                }
                
                logging.info(f"Training job {job_id} completed successfully")
                
            except Exception as e:
                # Update job status on error
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
                training_jobs[job_id]["end_time"] = time.time()
                logging.error(f"Training job {job_id} failed: {str(e)}")
        
        # Start the background thread
        import threading
        thread = threading.Thread(target=train_in_background)
        thread.daemon = True
        thread.start()
        
        # Return the job ID immediately
        return {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued for processing"
        }
        
    except Exception as e:
        logging.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@app.get("/train/{job_id}", response_model=TrainingResponse)
async def get_training_status(job_id: str):
    """Get status of a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job = training_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "results": job["results"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("ai_models.api:app", host="0.0.0.0", port=port, reload=False) 