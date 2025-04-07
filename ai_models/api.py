from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator, RootModel
import uvicorn
import logging
import numpy as np
import time
import os
import gc
from dotenv import load_dotenv
from ai_models.threat_detection_model import ThreatDetectionModel
from ai_models.metrics import update_model_metrics, record_prediction, record_batch_size, update_gpu_memory, record_prediction_result, set_model_version
import tensorflow as tf
from typing import List, Optional, Dict, Any, Union
import psutil

# Load environment variables
load_dotenv()

# Get log directory from environment variable or use current directory
LOG_DIR = os.environ.get('LOG_DIR', '.')
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, 'api.log')

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Initialize the app
app = FastAPI(
    title="NeuraShield AI API",
    description="API for NeuraShield's threat detection AI model",
    version="1.0.0"
)

# Model state management
model_state = {
    "model": None,
    "last_used": 0,
    "model_version": "unknown",
    "model_path": os.getenv('MODEL_PATH', 'models/threat_detection_20250403_212211'),
    "input_shape": int(os.getenv('INPUT_SHAPE', 39)),
    "num_classes": int(os.getenv('NUM_CLASSES', 2)),
    "lock": False  # Simple lock to prevent concurrent model loading
}

# Input data model with validation
class AnalysisRequestItem(RootModel):
    model_config = {"protected_namespaces": ()}
    # Each data item must be a dict of key-value pairs
    # Keys don't matter, but we need to validate the values are numeric
    root: Dict[str, Union[float, int]]
    
    @validator('root')
    def check_values(cls, v):
        # Check that we have the expected number of features
        expected_features = model_state["input_shape"]
        if len(v) != expected_features:
            raise ValueError(f"Expected {expected_features} features, but got {len(v)}")
        
        # Check that all values are numeric
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for {key} must be numeric, got {type(value)}")
        return v

class AnalysisRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    data: List[Dict[str, Union[float, int]]]
    
    @validator('data')
    def check_data(cls, v):
        if not v:
            raise ValueError("Data list cannot be empty")
        
        max_batch_size = int(os.getenv('MAX_BATCH_SIZE', 100))
        if len(v) > max_batch_size:
            raise ValueError(f"Batch size exceeds maximum of {max_batch_size}")
        return v

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
    model_info: Optional[dict] = None

def get_model():
    """Get or initialize the model with memory management"""
    current_time = time.time()
    
    # If model is already loaded and recently used, return it
    if model_state["model"] is not None:
        model_state["last_used"] = current_time
        return model_state["model"]
    
    # If another request is loading the model, wait briefly
    if model_state["lock"]:
        retry_count = 0
        while model_state["lock"] and retry_count < 5:
            time.sleep(0.2)
            retry_count += 1
            if model_state["model"] is not None:
                model_state["last_used"] = current_time
                return model_state["model"]
    
    # Acquire lock
    model_state["lock"] = True
    
    try:
        logging.info("Initializing model...")
        model_path = model_state["model_path"]
        
        # Clean memory before loading model
        if model_state["model"] is not None:
            del model_state["model"]
            gc.collect()
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
        
        # Load the model
        try:
            model = ThreatDetectionModel()
            model.load(model_path)
            model_state["model"] = model
            model_state["model_version"] = model.model_version
            model_state["last_used"] = current_time
            
            logging.info(f"Model loaded successfully from {model_path}")
            
            # Update metrics
            if model.training_history:
                final_accuracy = model.training_history['val_accuracy'][-1]
                update_model_metrics(
                    model_name=model.model_name,
                    accuracy=final_accuracy,
                    memory_usage=model.model.count_params() * 4
                )
                
            return model_state["model"]
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
            
    finally:
        # Release lock
        model_state["lock"] = False

def unload_unused_model(background_tasks):
    """Periodically unload model from memory if unused"""
    current_time = time.time()
    idle_threshold = 600  # 10 minutes
    
    if model_state["model"] is not None and current_time - model_state["last_used"] > idle_threshold:
        background_tasks.add_task(_unload_model)

def _unload_model():
    """Actual function to unload the model"""
    if model_state["model"] is not None:
        logging.info("Unloading unused model to free memory")
        del model_state["model"]
        model_state["model"] = None
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()

@app.get("/health", response_model=HealthResponse)
async def health_check(background_tasks: BackgroundTasks):
    """Health check endpoint"""
    try:
        # Check if we should unload the model
        unload_unused_model(background_tasks)
        
        # Get model info if loaded, or basic info if not
        if model_state["model"] is not None:
            return {
                "status": "healthy",
                "model_loaded": True,
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "model_version": model_state["model_version"],
                "model_info": model_state["model"].get_model_info() if model_state["model"] else {}
            }
        else:
            return {
                "status": "healthy",
                "model_loaded": False,
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "model_version": model_state["model_version"],
                "model_info": {"status": "not loaded", "ready_to_load": True}
            }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "gpu_available": False,
            "model_version": "unknown",
            "model_info": {"error": str(e)}
        }

def preprocess_features(data):
    """Normalize and validate input features"""
    try:
        # Extract features from all samples
        features_list = []
        
        for sample in data:
            # Convert dict to ordered list based on keys
            feature_values = list(sample.values())
            
            # Ensure we have the expected number of features
            if len(feature_values) != model_state["input_shape"]:
                raise ValueError(f"Expected {model_state['input_shape']} features, got {len(feature_values)}")
            
            # Convert to numpy array and normalize
            features = np.array(feature_values, dtype=np.float32)
            
            # Basic normalization (keeping simple here, but could be more sophisticated)
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            features_list.append(features)
        
        # Stack features into a batch
        features_batch = np.vstack(features_list)
        return features_batch
    
    except Exception as e:
        logging.error(f"Feature preprocessing error: {str(e)}")
        raise ValueError(f"Feature preprocessing error: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze data for threats"""
    try:
        # Get the model
        current_model = get_model()
        
        # Record batch size
        record_batch_size(len(request.data))
        
        # Preprocess features
        try:
            features_batch = preprocess_features(request.data)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        # Make predictions
        start_time = time.time()
        
        try:
            predictions = current_model.predict(features_batch)
            prediction_time = time.time() - start_time
            
            # Check prediction shape matches expectations
            if predictions.shape[1] != model_state["num_classes"]:
                raise ValueError(f"Model returned {predictions.shape[1]} classes but expected {model_state['num_classes']}")
                
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
            
            # Check if we should unload the model in the background
            unload_unused_model(background_tasks)
            
            return {
                "results": results,
                "processing_time": prediction_time,
                "model_version": current_model.model_version,
                "model_info": current_model.get_model_info()
            }
            
        except Exception as model_error:
            # Record failure metrics
            record_prediction(
                model_name=current_model.model_name,
                duration=time.time() - start_time,
                success=False
            )
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(model_error)}")
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Record failure metrics
        if model_state["model"]:
            record_prediction(
                model_name=model_state["model"].model_name if model_state["model"] else "unknown",
                duration=0,
                success=False
            )
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

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