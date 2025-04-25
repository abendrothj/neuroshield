"""
Metrics collection module for the NeuraShield threat detection system.
"""

import os
import time
import threading
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Get log directory from environment variable or use current directory
LOG_DIR = os.environ.get('LOG_DIR', './logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'ai_metrics.log')

# Initialize metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the model', ['model_name'])
MODEL_MEMORY_USAGE = Gauge('model_memory_usage', 'Memory usage of the model in bytes', ['model_name'])
PREDICTION_DURATION = Histogram('prediction_duration', 'Time taken for prediction in seconds', ['model_name'])
PREDICTION_COUNT = Counter('prediction_count', 'Number of predictions made', ['model_name', 'success'])
BATCH_SIZE = Histogram('batch_size', 'Size of prediction batches')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage', 'GPU memory usage in bytes', ['device'])
PREDICTION_RESULT = Counter('prediction_result', 'Distribution of prediction results', ['result'])
MODEL_VERSION = Gauge('model_version', 'Model version and deployment time', ['version', 'deployment_time'])

# Flag to track if metrics server is started
server_started = False
server_lock = threading.Lock()

def start_metrics_server(port=8000):
    """Start Prometheus metrics server"""
    global server_started
    
    with server_lock:
        if not server_started:
            try:
        start_http_server(port)
                server_started = True
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Metrics server started on port {port}\n")
    except Exception as e:
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error starting metrics server: {str(e)}\n")

def update_model_metrics(model_name, accuracy, memory_usage):
    """Update model performance metrics"""
    MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
    MODEL_MEMORY_USAGE.labels(model_name=model_name).set(memory_usage)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Updated metrics for model {model_name}: accuracy={accuracy}, memory={memory_usage}\n")

def record_prediction(model_name, duration, success):
    """Record a prediction event"""
    PREDICTION_DURATION.labels(model_name=model_name).observe(duration)
    PREDICTION_COUNT.labels(model_name=model_name, success=str(success)).inc()
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Prediction recorded: model={model_name}, duration={duration}, success={success}\n")

def record_batch_size(size):
    """Record batch size for predictions"""
    BATCH_SIZE.observe(size)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Batch size recorded: {size}\n")

def update_gpu_memory():
    """Update GPU memory usage metrics if GPU is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            for i, gpu in enumerate(gpus):
                # Try to get memory info if available
                try:
                    gpu_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    if gpu_info and 'current' in gpu_info:
                        GPU_MEMORY_USAGE.labels(device=f"gpu_{i}").set(gpu_info['current'])
                        with open(LOG_FILE, 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - GPU {i} memory usage: {gpu_info['current']} bytes\n")
    except:
                    pass
    except Exception as e:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error updating GPU metrics: {str(e)}\n")

def record_prediction_result(result, confidence=0.0):
    """Record prediction result type"""
    PREDICTION_RESULT.labels(result=result).inc()
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Prediction result: {result} (confidence: {confidence})\n")

def set_model_version(version, deployment_time):
    """Set model version information"""
    MODEL_VERSION.labels(version=version, deployment_time=deployment_time).set(1)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Model version: {version}, deployment time: {deployment_time}\n") 