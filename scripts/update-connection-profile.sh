#!/bin/bash

# This script updates the Fabric connection profile to use Kubernetes service names
# instead of localhost, for use in a Kubernetes environment.

# Exit on error
set -e

CONNECTION_PROFILE_PATH="../backend/connection-profile.json"
K8S_CONNECTION_PROFILE_PATH="../backend/connection-profile-k8s.json"

echo "Updating connection profile for Kubernetes environment..."

# Create a new connection profile for Kubernetes
cat "$CONNECTION_PROFILE_PATH" | \
  sed 's|"url": "grpcs://localhost:7051"|"url": "grpcs://fabric-peer:7051"|g' | \
  sed 's|"url": "https://localhost:7054"|"url": "https://fabric-ca:7054"|g' | \
  sed 's|"path": "../fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"|"path": "/etc/hyperledger/fabric/tls/ca.crt"|g' | \
  sed 's|"path": "../fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/ca/ca.org1.example.com-cert.pem"|"path": "/etc/hyperledger/fabric-ca-server/ca-cert.pem"|g' \
  > "$K8S_CONNECTION_PROFILE_PATH"

echo "Connection profile updated for Kubernetes. Saved at $K8S_CONNECTION_PROFILE_PATH"
echo "Use this connection profile when deploying to Kubernetes." 