#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Setting up NeuraShield local development environment...${NC}"

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo -e "${RED}Minikube is not installed. Please install it first.${NC}"
    echo "Visit: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Kubectl is not installed. Please install it first.${NC}"
    echo "Visit: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Start minikube if not running
if ! minikube status | grep -q "Running"; then
    echo -e "${YELLOW}Starting minikube...${NC}"
    minikube start --cpus=4 --memory=8192 --disk-size=20g
    echo -e "${GREEN}Minikube started successfully!${NC}"
else
    echo -e "${GREEN}Minikube is already running.${NC}"
fi

# Enable required addons
echo -e "${YELLOW}Enabling required addons...${NC}"
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable dashboard

# Set up local DNS
echo -e "${YELLOW}Setting up local DNS...${NC}"
echo "127.0.0.1 neurashield.local" | sudo tee -a /etc/hosts

# Get minikube IP
MINIKUBE_IP=$(minikube ip)
echo -e "${GREEN}Minikube IP: ${MINIKUBE_IP}${NC}"

# Update hosts file with minikube IP
echo "${MINIKUBE_IP} neurashield.local" | sudo tee -a /etc/hosts

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
eval $(minikube docker-env)

# Build frontend image
echo -e "${YELLOW}Building frontend image...${NC}"
docker build -t neurashield-frontend:latest -f Dockerfile.frontend .

# Build backend image
echo -e "${YELLOW}Building backend image...${NC}"
docker build -t neurashield-backend:latest -f Dockerfile.backend .

# Build AI service image
echo -e "${YELLOW}Building AI service image...${NC}"
docker build -t neurashield-ai:latest -f Dockerfile.ai .

echo -e "${GREEN}Local development environment setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run './k8s/deploy.sh' to deploy the application"
echo "2. Access the application at http://neurashield.local"
echo "3. Access the Kubernetes dashboard with: minikube dashboard" 