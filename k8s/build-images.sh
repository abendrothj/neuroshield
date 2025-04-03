#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Building Docker images for NeuraShield...${NC}"

# Build frontend image
echo "Building frontend image..."
docker build -t neurashield-frontend:latest \
    -f ${PROJECT_ROOT}/frontend/Dockerfile \
    ${PROJECT_ROOT}/frontend

# Build backend image
echo "Building backend image..."
docker build -t neurashield-backend:latest \
    -f ${PROJECT_ROOT}/backend/Dockerfile \
    ${PROJECT_ROOT}/backend

# Build AI service image
echo "Building AI service image..."
docker build -t neurashield-ai:latest \
    -f ${PROJECT_ROOT}/ai_models/Dockerfile \
    ${PROJECT_ROOT}/ai_models

echo -e "${GREEN}All images built successfully!${NC}"

# List built images
echo -e "${YELLOW}Built images:${NC}"
docker images | grep neurashield 