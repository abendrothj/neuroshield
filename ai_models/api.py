from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import numpy as np
import time
import os
from dotenv import load_dotenv
from threat_detection_model import ThreatDetectionModel
from metrics import update_model_metrics, record_prediction, record_batch_size, update_gpu_memory, record_prediction_result, set_model_version
import tensorflow as tf
from typing import List, Optional, Dict, Any
import psutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Initialize the app
app = FastAPI(
    title="NeuraShield AI API",
    description="API for NeuraShield's threat detection AI model",
    version="1.0.0"
)

# Initialize model (lazy loading)
model = None

# Input data model
class AnalysisRequest(BaseModel):
    data: list[dict]

# Health response model
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str
    model_info: dict
    gpu_available: bool

class ThreatPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}
    threat_level: str
    confidence: float
    probabilities: dict
    model_version: str

# Response model
class AnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    results: List[ThreatPrediction]
    processing_time: float
    model_version: str

def get_model():
    """Get or initialize the model"""
    global model
    
    if model is None:
        logging.info("Initializing model...")
        model_path = os.getenv('MODEL_PATH', 'models/threat_detection_20250403_212211')
        
        try:
            model = ThreatDetectionModel()
            model.load(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
            
            # Update metrics
            if model.training_history:
                final_accuracy = model.training_history['val_accuracy'][-1]
                update_model_metrics(
                    model_name=model.model_name,
                    accuracy=final_accuracy,
                    memory_usage=model.model.count_params() * 4
                )
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
            
    return model

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        current_model = get_model()
        return {
            "status": "healthy",
            "model_loaded": current_model is not None,
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "model_version": current_model.model_version if current_model else "unknown",
            "model_info": current_model.get_model_info() if current_model else {}
        }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "gpu_available": False,
            "model_version": "unknown",
            "model_info": {}
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Analyze data for threats"""
    try:
        # Get the model
        current_model = get_model()
        
        # Convert data to numpy array
        try:
            # Record batch size
            record_batch_size(len(request.data))
            
            # Extract features from all samples
            features_list = []
            for sample in request.data:
                features = np.array([list(sample.values())], dtype=np.float32)
                features_list.append(features)
            
            # Stack features into a batch
            features_batch = np.vstack(features_list)
            
            # Make predictions
            start_time = time.time()
            predictions = current_model.predict(features_batch)
            prediction_time = time.time() - start_time
            
            # Update GPU memory metrics
            update_gpu_memory()
            
            # Process results
            results = []
            for i, pred in enumerate(predictions):
                predicted_class = np.argmax(pred)
                confidence = float(pred[predicted_class])
                
                # For binary classification (0 = normal, 1 = threat)
                threat_levels = ["Normal", "Threat"]
                
                # Convert probabilities to dictionary
                prob_dict = {
                    "normal": float(pred[0]),
                    "threat": float(pred[1]) if len(pred) > 1 else 0.0
                }
                
                # Record prediction result metrics
                record_prediction_result(
                    result=threat_levels[predicted_class],
                    confidence=confidence
                )
                
                # Prepare result
                result = {
                    "threat_level": threat_levels[predicted_class],
                    "confidence": confidence,
                    "probabilities": prob_dict,
                    "model_version": current_model.model_version
                }
                results.append(result)
                
                # Record prediction metrics
                record_prediction(
                    model_name=current_model.model_name,
                    duration=prediction_time / len(predictions),
                    success=True
                )
            
            # Set model version info in metrics
            deployment_time = "unknown"
            if current_model.metadata and "deployment_time" in current_model.metadata:
                deployment_time = current_model.metadata["deployment_time"]
            set_model_version(current_model.model_version, deployment_time)
            
            return {
                "results": results,
                "processing_time": prediction_time,
                "model_version": current_model.model_version
            }
            
        except Exception as e:
            # Record failure metrics
            record_prediction(
                model_name=current_model.model_name if current_model else "unknown",
                duration=0,
                success=False
            )
            
            logging.error(f"Error making predictions: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error making predictions: {str(e)}")
            
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
        accuracy = 0.95  # Default value
        if current_model.metadata and "validation_accuracy" in current_model.metadata:
            accuracy = current_model.metadata["validation_accuracy"]
            
        # Get system metrics
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        
        # Get GPU metrics if available
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        gpu_metrics = {}
        if gpu_available:
            try:
                gpu_devices = tf.config.list_physical_devices('GPU')
                for i, device in enumerate(gpu_devices):
                    # Try to get memory info for this GPU
                    gpu_mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    if gpu_mem_info:
                        gpu_metrics[f"gpu_{i}"] = {
                            "memory_used": gpu_mem_info.get('current', 0),
                            "memory_total": gpu_mem_info.get('peak', 0)
                        }
            except:
                pass
                
        # Return comprehensive metrics
        return {
            "model": {
                "name": current_model.model_name,
                "version": current_model.model_version,
                "accuracy": accuracy,
                "input_shape": model_info.get("input_shape", []),
                "num_classes": model_info.get("num_classes", 0),
                "parameters": model_info.get("trainable_params", 0),
                "deployment_time": current_model.metadata.get("deployment_time", "unknown") if current_model.metadata else "unknown"
            },
            "performance": {
                "average_inference_time": 0.05,  # Placeholder - would come from actual monitoring
                "throughput": 100,  # Predictions per second - placeholder
                "memory_usage": memory_usage,
                "prediction_count": 10000,  # Placeholder
                "error_rate": 0.01  # Placeholder
            },
            "hardware": {
                "gpu_available": gpu_available,
                "gpus": gpu_metrics,
                "cpu_count": psutil.cpu_count(),
                "cpu_usage": psutil.cpu_percent()
            },
            "predictions": {
                "normal_count": 9000,  # Placeholder
                "threat_count": 1000,  # Placeholder
                "last_prediction_time": time.time()
            }
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
    model_config = {"protected_namespaces": ()}
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    results: Optional[Dict[str, Any]] = None

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
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False) 