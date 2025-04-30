#!/usr/bin/env python3
"""
TensorFlow GPU Test Script for NeuraShield container
This script validates that TensorFlow can access the GPU inside the Docker container
"""

import os
import tensorflow as tf
import logging
import psutil
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def test_gpu():
    """Test if TensorFlow can access the GPU"""
    logging.info("Testing TensorFlow GPU support...")
    
    # Check if TensorFlow was built with CUDA support
    cuda_available = tf.test.is_built_with_cuda()
    logging.info(f"TensorFlow built with CUDA: {cuda_available}")
    
    # Check physical devices
    physical_devices = tf.config.list_physical_devices()
    logging.info(f"All physical devices: {physical_devices}")
    
    # Check GPU devices specifically
    gpus = tf.config.list_physical_devices('GPU')
    logging.info(f"GPU devices: {gpus}")
    
    if not gpus:
        logging.warning("No GPU devices detected!")
        return False
    
    # Try to run a simple operation on GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            logging.info(f"Matrix multiplication result: {c}")
            logging.info(f"Operation performed on device: {c.device}")
            
            # Check if operation was actually performed on GPU
            if 'GPU' in c.device:
                logging.info("Successfully ran operations on GPU!")
                return True
            else:
                logging.warning("Operation was performed on CPU despite GPU being available")
                return False
    except Exception as e:
        logging.error(f"Error running GPU operation: {str(e)}")
        return False

def log_system_info():
    """Log system information"""
    logging.info("=== System Information ===")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"Python version: {os.sys.version}")
    
    # Log CPU information
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=1)
    logging.info(f"CPU: {cpu_count} physical cores, {cpu_count_logical} logical cores, Usage: {cpu_percent}%")
    
    # Log memory information
    memory = psutil.virtual_memory()
    logging.info(f"Memory: Total: {memory.total / (1024**3):.2f} GB, Available: {memory.available / (1024**3):.2f} GB, Used: {memory.percent}%")
    
    # Try to get GPU information using NVIDIA management library
    try:
        nvidia_smi_output = os.popen('nvidia-smi').read()
        logging.info(f"NVIDIA SMI Output:\n{nvidia_smi_output}")
    except Exception as e:
        logging.warning(f"Could not get NVIDIA SMI output: {str(e)}")

def benchmark_gpu():
    """Run a simple benchmark to test GPU performance"""
    logging.info("Running GPU benchmark...")
    
    try:
        # Create a large tensor operation to benchmark
        n = 5000
        
        # CPU timing
        start_time = time.time()
        with tf.device('/CPU:0'):
            a_cpu = tf.random.normal([n, n])
            b_cpu = tf.random.normal([n, n])
            c_cpu = tf.matmul(a_cpu, b_cpu)
            # Force evaluation
            result_cpu = c_cpu.numpy()
        cpu_time = time.time() - start_time
        logging.info(f"CPU computation time: {cpu_time:.4f} seconds")
        
        # GPU timing (if available)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Clear memory
            tf.keras.backend.clear_session()
            
            start_time = time.time()
            with tf.device('/GPU:0'):
                a_gpu = tf.random.normal([n, n])
                b_gpu = tf.random.normal([n, n])
                c_gpu = tf.matmul(a_gpu, b_gpu)
                # Force evaluation
                result_gpu = c_gpu.numpy()
            gpu_time = time.time() - start_time
            logging.info(f"GPU computation time: {gpu_time:.4f} seconds")
            
            # Compare performance
            if cpu_time > 0 and gpu_time > 0:
                speedup = cpu_time / gpu_time
                logging.info(f"GPU is {speedup:.2f}x faster than CPU")
        else:
            logging.warning("No GPU available for benchmarking")
    
    except Exception as e:
        logging.error(f"Error during benchmarking: {str(e)}")

if __name__ == "__main__":
    logging.info("=== NeuraShield GPU Test ===")
    log_system_info()
    gpu_available = test_gpu()
    if gpu_available:
        benchmark_gpu()
    logging.info("=== Test Complete ===") 