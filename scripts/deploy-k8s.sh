#!/bin/bash
set -e

# Set variables
PROJECT_ID="supple-defender-458307-i7"
REGION="us-west1"
CLUSTER_NAME="neurashield-cluster"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building and deploying NeuraShield to GKE in ${REGION}...${NC}"

# Ensure we're connected to the right cluster
echo -e "${GREEN}Connecting to GKE cluster...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION}

# Build and push backend
echo -e "${GREEN}Building backend image...${NC}"
cd "$(dirname "$0")/../backend"
docker build -t gcr.io/${PROJECT_ID}/neurashield-backend:latest .
echo -e "${GREEN}Pushing backend image to GCR...${NC}"
docker push gcr.io/${PROJECT_ID}/neurashield-backend:latest

# Build and push frontend
echo -e "${GREEN}Building frontend image...${NC}"
cd "$(dirname "$0")/../frontend"
docker build -t gcr.io/${PROJECT_ID}/neurashield-frontend:latest .
echo -e "${GREEN}Pushing frontend image to GCR...${NC}"
docker push gcr.io/${PROJECT_ID}/neurashield-frontend:latest

# Build and push blockchain
echo -e "${GREEN}Building blockchain image...${NC}"
cd "$(dirname "$0")/../blockchain"
docker build -t gcr.io/${PROJECT_ID}/neurashield-blockchain:latest .
echo -e "${GREEN}Pushing blockchain image to GCR...${NC}"
docker push gcr.io/${PROJECT_ID}/neurashield-blockchain:latest

# Apply Kubernetes manifests
echo -e "${GREEN}Applying Kubernetes manifests...${NC}"
cd "$(dirname "$0")/.."
kubectl apply -f k8s/neurashield-k8s.yaml

# Wait for deployments to be ready
echo -e "${GREEN}Waiting for deployments to be ready...${NC}"
kubectl wait --namespace neurashield --for=condition=available deployments --all --timeout=300s

# Get services information
echo -e "${GREEN}Services deployed successfully:${NC}"
kubectl get services -n neurashield

# Get frontend service external IP
FRONTEND_IP=$(kubectl get service neurashield-frontend -n neurashield -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$FRONTEND_IP" ]; then
  echo -e "${GREEN}NeuraShield frontend is available at: http://${FRONTEND_IP}${NC}"
else
  echo -e "${YELLOW}Waiting for frontend LoadBalancer IP to be assigned...${NC}"
  echo -e "${YELLOW}Run: kubectl get service neurashield-frontend -n neurashield${NC}"
fi

echo -e "${GREEN}Deployment completed successfully!${NC}" 