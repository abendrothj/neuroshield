from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary, REGISTRY
import time
import threading
import logging
import os

# Get log directory from environment variable or use current directory
LOG_DIR = os.environ.get('LOG_DIR', '.')
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, 'ai_metrics.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Helper function to check if a metric already exists
def metric_exists(name):
    # Safely check if a metric name already exists in the registry
    return name in REGISTRY._names_to_collectors

# Create metrics only if they don't already exist
if not metric_exists('model_predictions_total'):
    model_predictions_total = Counter(
        'model_predictions_total',
        'Total number of model predictions',
        ['model_name', 'status']
    )
else:
    # Get existing metric
    model_predictions_total = REGISTRY._names_to_collectors['model_predictions_total']

if not metric_exists('model_prediction_duration_seconds'):
    model_prediction_duration = Histogram(
        'model_prediction_duration_seconds',
        'Duration of model predictions in seconds',
        ['model_name'],
        buckets=[0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
    )
else:
    model_prediction_duration = REGISTRY._names_to_collectors['model_prediction_duration_seconds']

if not metric_exists('model_accuracy'):
    model_accuracy = Gauge(
        'model_accuracy',
        'Current accuracy of the threat detection model',
        ['model_name']
    )
else:
    model_accuracy = REGISTRY._names_to_collectors['model_accuracy']

if not metric_exists('model_memory_usage_bytes'):
    model_memory_usage = Gauge(
        'model_memory_usage_bytes',
        'Memory usage of the model in bytes',
        ['model_name']
    )
else:
    model_memory_usage = REGISTRY._names_to_collectors['model_memory_usage_bytes']

if not metric_exists('model_gpu_utilization_percent'):
    model_gpu_utilization = Gauge(
        'model_gpu_utilization_percent',
        'GPU utilization percentage',
        ['gpu_id']
    )
else:
    model_gpu_utilization = REGISTRY._names_to_collectors['model_gpu_utilization_percent']

if not metric_exists('model_version_info'):
    model_version = Gauge(
        'model_version_info',
        'Information about the currently loaded model version',
        ['version', 'deployment_time']
    )
else:
    model_version = REGISTRY._names_to_collectors['model_version_info']

if not metric_exists('prediction_requests_total'):
    prediction_requests_total = Counter(
        'prediction_requests_total', 
        'Total number of prediction requests',
        ['status']
    )
else:
    prediction_requests_total = REGISTRY._names_to_collectors['prediction_requests_total']

if not metric_exists('prediction_results_total'):
    prediction_results_total = Counter(
        'prediction_results_total', 
        'Total number of predictions by result',
        ['result']
    )
else:
    prediction_results_total = REGISTRY._names_to_collectors['prediction_results_total']

if not metric_exists('prediction_latency_seconds'):
    prediction_latency = Histogram(
        'prediction_latency_seconds', 
        'Prediction request latency in seconds',
        ['status'],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
    )
else:
    prediction_latency = REGISTRY._names_to_collectors['prediction_latency_seconds']

if not metric_exists('prediction_confidence'):
    prediction_confidence = Summary(
        'prediction_confidence',
        'Confidence of predictions',
        ['result']
    )
else:
    prediction_confidence = REGISTRY._names_to_collectors['prediction_confidence']

if not metric_exists('gpu_memory_bytes'):
    gpu_memory_usage = Gauge(
        'gpu_memory_bytes',
        'GPU memory usage in bytes'
    )
else:
    gpu_memory_usage = REGISTRY._names_to_collectors['gpu_memory_bytes']

if not metric_exists('batch_size'):
    batch_size = Summary(
        'batch_size',
        'Size of prediction batches'
    )
else:
    batch_size = REGISTRY._names_to_collectors['batch_size']

if not metric_exists('feature_importance'):
    feature_importance = Gauge(
        'feature_importance',
        'Importance of each feature in the model',
        ['feature_name']
    )
else:
    feature_importance = REGISTRY._names_to_collectors['feature_importance']

def start_metrics_server(port=8000):
    """Start the Prometheus metrics server"""
    try:
        # Check if the server is already running
        if hasattr(start_metrics_server, 'started') and start_metrics_server.started:
            logging.info(f'Metrics server already running on port {port}')
            return
            
        start_http_server(port)
        start_metrics_server.started = True
        logging.info(f'Metrics server started on port {port}')
    except Exception as e:
        logging.error(f'Failed to start metrics server: {str(e)}')
        raise

# Set the initial state
start_metrics_server.started = False

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