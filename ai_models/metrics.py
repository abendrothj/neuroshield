from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import time
import threading
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_metrics.log'),
        logging.StreamHandler()
    ]
)

# Create metrics
model_predictions_total = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'status']
)

model_prediction_duration = Histogram(
    'model_prediction_duration_seconds',
    'Duration of model predictions in seconds',
    ['model_name'],
    buckets=[0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current accuracy of the threat detection model',
    ['model_name']
)

model_memory_usage = Gauge(
    'model_memory_usage_bytes',
    'Memory usage of the model in bytes',
    ['model_name']
)

model_gpu_utilization = Gauge(
    'model_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

model_version = Gauge(
    'model_version_info',
    'Information about the currently loaded model version',
    ['version', 'deployment_time']
)

prediction_requests_total = Counter(
    'prediction_requests_total', 
    'Total number of prediction requests',
    ['status']
)

prediction_results_total = Counter(
    'prediction_results_total', 
    'Total number of predictions by result',
    ['result']
)

prediction_latency = Histogram(
    'prediction_latency_seconds', 
    'Prediction request latency in seconds',
    ['status'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
)

prediction_confidence = Summary(
    'prediction_confidence',
    'Confidence of predictions',
    ['result']
)

gpu_memory_usage = Gauge(
    'gpu_memory_bytes',
    'GPU memory usage in bytes'
)

batch_size = Summary(
    'batch_size',
    'Size of prediction batches'
)

feature_importance = Gauge(
    'feature_importance',
    'Importance of each feature in the model',
    ['feature_name']
)

def start_metrics_server(port=8000):
    """Start the Prometheus metrics server"""
    try:
        start_http_server(port)
        logging.info(f'Metrics server started on port {port}')
    except Exception as e:
        logging.error(f'Failed to start metrics server: {str(e)}')
        raise

def update_model_metrics(model_name, accuracy, memory_usage, gpu_utilization=None):
    """Update model-related metrics"""
    try:
        model_accuracy.labels(model_name=model_name).set(accuracy)
        model_memory_usage.labels(model_name=model_name).set(memory_usage)
        
        if gpu_utilization is not None:
            for gpu_id, utilization in enumerate(gpu_utilization):
                model_gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization)
                
    except Exception as e:
        logging.error(f'Failed to update model metrics: {str(e)}')
        raise

def record_prediction(model_name, duration, success=True):
    """Record a model prediction"""
    try:
        status = 'success' if success else 'error'
        model_predictions_total.labels(
            model_name=model_name,
            status=status
        ).inc()
        
        model_prediction_duration.labels(
            model_name=model_name
        ).observe(duration)
        
    except Exception as e:
        logging.error(f'Failed to record prediction: {str(e)}')
        raise

def record_prediction_result(result, confidence):
    """Record a prediction result"""
    prediction_results_total.labels(result=result).inc()
    prediction_confidence.labels(result=result).observe(confidence)

def set_model_version(version, deployment_time):
    """Set model version info"""
    model_version.labels(version=version, deployment_time=deployment_time).set(1)

def update_gpu_memory():
    """Update GPU memory usage"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Get GPU memory usage (this will only work with TensorFlow)
            gpu_mem_info = tf.config.experimental.get_memory_info('GPU:0')
            gpu_memory_usage.set(gpu_mem_info.get('current', 0))
    except:
        # If there's an error, just set to 0
        gpu_memory_usage.set(0)
        
def record_batch_size(size):
    """Record batch size"""
    batch_size.observe(size) 