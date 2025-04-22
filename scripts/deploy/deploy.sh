#!/bin/bash

# NeuraShield Deployment Script
# This script handles deployment of all components including chaincode and Kubernetes

set -e

# Configuration
BLOCKCHAIN_DIR="blockchain"
BACKEND_DIR="backend"
K8S_DIR="k8s"
CHANNEL_NAME="neurashield-channel"
CHAINCODE_NAME="neurashield"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Print with color
print() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Deploy blockchain network
deploy_blockchain() {
    print "Deploying blockchain network..."
    
    cd $BLOCKCHAIN_DIR
    
    # Start the network
    ./network.sh up createChannel -c $CHANNEL_NAME -ca
    
    # Deploy chaincode
    ./network.sh deployCC -ccn $CHAINCODE_NAME -ccp ../$BACKEND_DIR/chaincode -ccl go -c $CHANNEL_NAME
    
    cd ..
    
    print "Blockchain deployment complete"
}

# Deploy backend services
deploy_backend() {
    print "Deploying backend services..."
    
    cd $BACKEND_DIR
    
    # Build Docker images
    docker-compose build
    
    # Start services
    docker-compose up -d
    
    cd ..
    
    print "Backend deployment complete"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    print "Deploying to Kubernetes..."
    
    cd $K8S_DIR
    
    # Apply configurations
    kubectl apply -f namespace.yaml
    kubectl apply -f configmaps.yaml
    kubectl apply -f secrets.yaml
    kubectl apply -f deployments.yaml
    kubectl apply -f services.yaml
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/neurashield-api -n neurashield
    kubectl wait --for=condition=available --timeout=300s deployment/neurashield-dashboard -n neurashield
    
    cd ..
    
    print "Kubernetes deployment complete"
}

# Update chaincode
update_chaincode() {
    print "Updating chaincode..."
    
    cd $BLOCKCHAIN_DIR
    
    # Package new chaincode
    peer lifecycle chaincode package neurashield.tar.gz --path ../$BACKEND_DIR/chaincode --lang golang --label neurashield_1.0
    
    # Install chaincode
    peer lifecycle chaincode install neurashield.tar.gz
    
    # Approve chaincode
    peer lifecycle chaincode approveformyorg -o orderer.example.com:7050 --channelID $CHANNEL_NAME --name $CHAINCODE_NAME --version 1.0 --package-id $(peer lifecycle chaincode queryinstalled | grep -o "neurashield_1.0:[a-z0-9]*") --sequence 1 --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem
    
    # Commit chaincode
    peer lifecycle chaincode commit -o orderer.example.com:7050 --channelID $CHANNEL_NAME --name $CHAINCODE_NAME --version 1.0 --sequence 1 --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses peer0.org1.example.com:7051 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    
    cd ..
    
    print "Chaincode update complete"
}

# Main deployment function
main() {
    print "Starting NeuraShield deployment..."
    
    # Check if running in Kubernetes mode
    if [ "$1" == "k8s" ]; then
        deploy_kubernetes
    else
        deploy_blockchain
        deploy_backend
    fi
    
    print "Deployment complete!"
}

# Run main function
main "$@" 