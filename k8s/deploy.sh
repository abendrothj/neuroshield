#!/bin/bash

# Exit on error
set -e

# Change to the script directory
cd "$(dirname "$0")"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Create namespace if it doesn't exist
echo "Creating namespace..."
kubectl apply -f namespace.yaml

# Apply ConfigMaps and Secrets
echo "Applying ConfigMaps and Secrets..."
kubectl apply -f configmap.yaml
kubectl apply -f fabric-configmaps.yaml
kubectl apply -f secrets.yaml

# Apply StorageClass and PVCs
echo "Applying Storage configurations..."
kubectl apply -f storage.yaml

# Deploy Blockchain components
echo "Deploying Hyperledger Fabric components..."
kubectl apply -f blockchain-deployment.yaml

# Wait for Fabric to be ready
echo "Waiting for Fabric pods to be ready..."
kubectl wait --for=condition=ready pod -l app=hyperledger --timeout=300s

# Deploy backend, frontend, and AI services
echo "Deploying application services..."
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml
kubectl apply -f ai-deployment.yaml

# Deploy Services
echo "Exposing services..."
kubectl apply -f services.yaml
kubectl apply -f backend-service.yaml
kubectl apply -f frontend-service.yaml
kubectl apply -f ai-service.yaml

# Apply Network Policies
echo "Applying network policies..."
kubectl apply -f network-policies.yaml

# Deploy Ingress
echo "Setting up ingress..."
kubectl apply -f ingress.yaml

# Setup monitoring
echo "Setting up monitoring..."
kubectl apply -f monitoring.yaml

# Apply backup CronJob
echo "Setting up backup job..."
kubectl apply -f backup-cronjob.yaml

echo "Deployment completed successfully!"
echo "You can access the services through the configured ingress." 