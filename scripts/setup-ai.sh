#!/bin/bash

# Exit on error
set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check for NVIDIA GPU and install necessary drivers/CUDA if needed
echo "Checking NVIDIA GPU and CUDA setup..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Installing NVIDIA drivers and CUDA..."
    sudo apt update
    sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
    echo "Please reboot your system after installation and run this script again."
    exit 0
else
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader
    echo "CUDA Version:"
    nvcc --version | grep "release" || echo "CUDA toolkit not found"
fi

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    sudo apt install -y nvidia-cuda-toolkit
fi

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Clean existing Python packages and cache
echo "Cleaning existing Python packages and cache..."
rm -rf ~/.cache/pip
rm -rf "$PROJECT_ROOT/ai_models/venv"
rm -rf "$PROJECT_ROOT/ai_models/__pycache__"
rm -rf "$PROJECT_ROOT/ai_models/*.pyc"
rm -rf "$PROJECT_ROOT/ai_models/*.pyo"
rm -rf "$PROJECT_ROOT/ai_models/*.pyd"
rm -rf "$PROJECT_ROOT/ai_models/.Python"
rm -rf "$PROJECT_ROOT/ai_models/build"
rm -rf "$PROJECT_ROOT/ai_models/dist"
rm -rf "$PROJECT_ROOT/ai_models/*.egg-info"

# Install Python 3.10 if not present
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
fi

# Create virtual environment with Python 3.10
echo "Creating Python virtual environment..."
python3.10 -m venv "$PROJECT_ROOT/ai_models/venv"

# Activate virtual environment
source "$PROJECT_ROOT/ai_models/venv/bin/activate"

# Verify Python version
echo "Python version:"
python --version

# Upgrade pip and install wheel
echo "Upgrading pip and installing wheel..."
pip install --upgrade pip
pip install wheel

# Install CUDA-specific packages first
echo "Installing CUDA-specific packages..."
pip install nvidia-cudnn-cu11==8.6.0.163
pip install tensorflow[and-cuda]==2.12.0

# Install other dependencies
echo "Installing remaining Python dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# Verify TensorFlow installation and check GPU
echo "Verifying TensorFlow installation and checking GPU..."
python -c "
import tensorflow as tf
import sys
import os

print('Python version:', sys.version)
print('TensorFlow version:', tf.__version__)

# Enable GPU memory growth and set memory limit for RTX 3060 Ti
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            # Enable memory growth
            tf.config.experimental.set_memory_growth(device, True)
            # Set memory limit to 8GB (8192MB) for RTX 3060 Ti
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
            )
        print('\nGPU configuration:')
        print('- Memory growth enabled')
        print('- Memory limit set to 8GB')
        print('- XLA compilation enabled')
        print('- Mixed precision enabled')
    except RuntimeError as e:
        print('\nGPU configuration error:', e)
else:
    print('\nNo GPU devices found in TensorFlow')

# Print detailed GPU information
print('\nTensorFlow GPU Information:')
print('Available GPUs:', tf.config.list_physical_devices('GPU'))
print('GPU Device Name:', tf.test.gpu_device_name())

# Check CUDA environment
print('\nCUDA Environment:')
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'Not set'))
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'Not set'))

# Configure optimizations based on GPU availability
if physical_devices:
    print('\nEnabling GPU optimizations...')
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision training enabled')
    
    # Test GPU with a simple computation
    print('\nTesting GPU with matrix multiplication...')
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print('GPU test completed successfully')
else:
    print('\nConfiguring CPU optimizations...')
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    print('CPU thread parallelism configured')
"

# Download pre-trained model (placeholder)
echo "Downloading pre-trained model..."
python -c "
import tensorflow as tf
from tensorflow import keras
import os

# Create models directory if it doesn't exist
os.makedirs('$PROJECT_ROOT/ai_models', exist_ok=True)

try:
    # Create a simple model for testing
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])

    # Enable mixed precision if GPU is available
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Compile the model with optimizations
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # Enable XLA compilation
    )

    # Save the model
    model.save('$PROJECT_ROOT/ai_models/threat_detection_model.h5')
    print('Model saved successfully')
except Exception as e:
    print('Error creating model:', str(e))
    sys.exit(1)
"

echo "AI environment setup completed successfully!" 