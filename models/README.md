# NeuraShield AI Container

This directory contains the containerized version of the NeuraShield AI threat detection service. The container packages all necessary components for AI-based threat detection with blockchain integration.

## Features

- TensorFlow-based threat detection model
- FastAPI-based REST API for inference
- Prometheus metrics collection
- Blockchain integration for secure logging
- GPU support (with NVIDIA Container Toolkit)
- Health checks and monitoring

## Prerequisites

- Docker Engine 20.10 or later
- NVIDIA Container Toolkit (for GPU support)
- At least 4GB RAM (8GB recommended)
- NVIDIA GPU with CUDA support (optional but recommended)

## Building the Container

```bash
# Build the container
docker build -f /home/jub/Cursor/neurashield/models/Dockerfile -t neurashield-ai:latest /home/jub/Cursor/neurashield
```

## Running the Container

### With GPU Support

```bash
docker run --gpus all \
    -p 5000:5000 -p 8000:8000 \
    -v /home/jub/Cursor/neurashield/models/data:/app/data \
    -v /home/jub/Cursor/neurashield/models/logs:/app/logs \
    -v /home/jub/Cursor/neurashield/models/models:/app/models \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    --name neurashield-ai \
    neurashield-ai:latest
```

### CPU Only

```bash
docker run \
    -p 5000:5000 -p 8000:8000 \
    -v /home/jub/Cursor/neurashield/models/data:/app/data \
    -v /home/jub/Cursor/neurashield/models/logs:/app/logs \
    -v /home/jub/Cursor/neurashield/models/models:/app/models \
    --name neurashield-ai \
    neurashield-ai:latest
```

## Using Docker Compose

For full environment with blockchain integration:

```bash
docker-compose up -d
```

## Environment Variables

The container supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API port | 5000 |
| `METRICS_PORT` | Prometheus metrics port | 8000 |
| `LOG_DIR` | Log directory | /app/logs |
| `MODEL_PATH` | Path to model file | /app/models/threat_detection_20250420_233903.keras |
| `BLOCKCHAIN_API_URL` | URL for blockchain API | http://blockchain-adapter:3000/api/v1/events |
| `BLOCKCHAIN_ENABLED` | Enable blockchain logging | true |
| `TF_FORCE_GPU_ALLOW_GROWTH` | TensorFlow GPU memory growth | true |

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /analyze` - Analyze data for threats
- `GET /metrics` - Prometheus metrics endpoint
- `POST /train` - Train model (async)
- `GET /train/{job_id}` - Get training status

## Testing GPU Support

The container includes a GPU test script:

```bash
# Inside the container
python /app/test_gpu_container.py
```

## Volumes

- `/app/data` - Data directory for model inputs
- `/app/logs` - Log directory
- `/app/models` - Directory for model files 