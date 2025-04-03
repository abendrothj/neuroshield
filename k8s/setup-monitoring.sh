#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setting up monitoring infrastructure...${NC}"

# Create monitoring namespace if it doesn't exist
echo "Creating monitoring namespace..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Install Prometheus Operator using Helm
echo "Installing Prometheus Operator using Helm..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Check if prometheus-operator is already installed
if helm list -n monitoring | grep -q "prometheus-operator"; then
    echo "Upgrading existing Prometheus Operator installation..."
    helm upgrade prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set grafana.enabled=true \
        --set prometheus.enabled=true \
        --set alertmanager.enabled=true
else
    echo "Installing new Prometheus Operator..."
    helm install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set grafana.enabled=true \
        --set prometheus.enabled=true \
        --set alertmanager.enabled=true
fi

# Wait for Prometheus Operator to be ready
echo "Waiting for Prometheus Operator to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus-operator-kube-p-operator -n monitoring

echo -e "${GREEN}Monitoring infrastructure setup complete!${NC}"

# Display Grafana access information
echo -e "${YELLOW}Grafana Access Information:${NC}"
echo "To get the Grafana admin password, run:"
echo "kubectl --namespace monitoring get secrets prometheus-operator-grafana -o jsonpath='{.data.admin-password}' | base64 -d"
echo ""
echo "To access Grafana, run:"
echo "kubectl --namespace monitoring port-forward svc/prometheus-operator-grafana 3000:80"
echo "Then visit http://localhost:3000 in your browser" 