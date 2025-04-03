import tensorflow as tf
import numpy as np
import os

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Print detailed system information
print("\nSystem Information:")
print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("GPU device name:", tf.test.gpu_device_name())
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

# Create a large matrix multiplication to test GPU
print("\nCreating test matrices...")
size = 5000
matrix1 = np.random.rand(size, size)
matrix2 = np.random.rand(size, size)

# Try to force GPU usage
print("\nAttempting GPU computation...")
try:
    with tf.device('/GPU:0'):
        print("Using GPU device")
        result = tf.matmul(matrix1, matrix2)
        print("Matrix multiplication completed on GPU")
        print("Result shape:", result.shape)
except RuntimeError as e:
    print("Error during GPU computation:", e)
    print("\nFalling back to CPU...")
    with tf.device('/CPU:0'):
        result = tf.matmul(matrix1, matrix2)
        print("Matrix multiplication completed on CPU")
        print("Result shape:", result.shape) 