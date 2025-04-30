#!/bin/bash

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NeuraShield AI Container Build and Run Script ===${NC}"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if nvidia-smi is available for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected!${NC}"
    echo -e "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Container will run in CPU-only mode.${NC}"
    GPU_AVAILABLE=false
fi

# Build the Docker image
echo -e "\n${GREEN}Building NeuraShield AI Docker image...${NC}"
docker build -f /home/jub/Cursor/neurashield/models/Dockerfile -t neurashield-ai:latest /home/jub/Cursor/neurashield

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Docker image built successfully!${NC}"

# Run the container
echo -e "\n${GREEN}Running NeuraShield AI container...${NC}"

# Create required directories if they don't exist
mkdir -p /home/jub/Cursor/neurashield/models/data
mkdir -p /home/jub/Cursor/neurashield/models/logs
mkdir -p /home/jub/Cursor/neurashield/models/models

# Run with GPU if available, otherwise CPU only
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}Running with GPU support...${NC}"
    docker run --rm --gpus all \
        -p 5000:5000 -p 8000:8000 \
        -v /home/jub/Cursor/neurashield/models/data:/app/data \
        -v /home/jub/Cursor/neurashield/models/logs:/app/logs \
        -v /home/jub/Cursor/neurashield/models/models:/app/models \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        --name neurashield-ai-container \
        neurashield-ai:latest
else
    echo -e "${YELLOW}Running in CPU-only mode...${NC}"
    docker run --rm \
        -p 5000:5000 -p 8000:8000 \
        -v /home/jub/Cursor/neurashield/models/data:/app/data \
        -v /home/jub/Cursor/neurashield/models/logs:/app/logs \
        -v /home/jub/Cursor/neurashield/models/models:/app/models \
        --name neurashield-ai-container \
        neurashield-ai:latest
fi 