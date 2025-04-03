#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Deploying NeuraShield to Kubernetes..."

# Build Docker images
echo "Building Docker images..."
${SCRIPT_DIR}/build-images.sh

# Check prerequisites
echo "Checking prerequisites..."
${SCRIPT_DIR}/check-prerequisites.sh

# Setup monitoring infrastructure
echo "Setting up monitoring infrastructure..."
${SCRIPT_DIR}/setup-monitoring.sh

# Create namespace
echo "Creating namespace..."
kubectl apply -f ${SCRIPT_DIR}/namespace.yaml

# Apply storage
echo "Applying storage..."
kubectl apply -f ${SCRIPT_DIR}/storage.yaml

# Apply secrets
echo "Applying secrets..."
kubectl apply -f ${SCRIPT_DIR}/secrets.yaml

# Apply network policies
echo "Applying network policies..."
kubectl apply -f ${SCRIPT_DIR}/network-policies.yaml

# Apply services
echo "Applying services..."
kubectl apply -f ${SCRIPT_DIR}/frontend-service.yaml
kubectl apply -f ${SCRIPT_DIR}/backend-service.yaml
kubectl apply -f ${SCRIPT_DIR}/ai-service.yaml

# Apply deployments
echo "Applying deployments..."
kubectl apply -f ${SCRIPT_DIR}/frontend-deployment.yaml
kubectl apply -f ${SCRIPT_DIR}/backend-deployment.yaml
kubectl apply -f ${SCRIPT_DIR}/ai-deployment.yaml

# Apply monitoring
echo "Applying monitoring..."
kubectl apply -f ${SCRIPT_DIR}/monitoring.yaml

# Apply ingress
echo "Applying ingress..."
kubectl apply -f ${SCRIPT_DIR}/ingress.yaml

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/neurashield-frontend -n neurashield
kubectl wait --for=condition=available --timeout=300s deployment/neurashield-backend -n neurashield
kubectl wait --for=condition=available --timeout=300s deployment/neurashield-ai -n neurashield

echo "Deployment complete!"
echo "Access the application at: http://neurashield.local"

# Show service URLs
echo -e "${YELLOW}Service URLs:${NC}"
echo "Frontend: http://neurashield.local"
echo "Backend API: http://neurashield.local/api"
echo "AI Service: http://neurashield.local/ai"

# Show pod status
echo -e "${YELLOW}Pod Status:${NC}"
kubectl get pods -l app=neurashield -n neurashield 