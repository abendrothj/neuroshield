from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import threading
import logging

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