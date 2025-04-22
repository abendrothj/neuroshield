#!/bin/bash

# Add CUDA paths to environment
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Enable memory growth for GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Print GPU information
echo "CUDA Version:"
nvcc --version
echo -e "\nGPU Information:"
nvidia-smi
echo -e "\nTensorFlow GPU Support:"
python3 -c "import tensorflow as tf; print('GPU Available:', tf.test.is_built_with_cuda()); print('GPU Device:', tf.test.gpu_device_name())" 