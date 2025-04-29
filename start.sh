#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== NeuraShield Startup Script =====${NC}"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
  exit 1
fi

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCKCHAIN_SCRIPTS="${SCRIPT_DIR}/scripts/blockchain"
SERVER_SCRIPTS="${SCRIPT_DIR}/scripts/server"

# Check Fabric network status
if ! docker ps | grep peer0.org1.example.com > /dev/null; then
  echo -e "${YELLOW}Fabric network is not running. Starting network...${NC}"
  bash "${BLOCKCHAIN_SCRIPTS}/setup-fabric-network.sh"
  
  # Deploy chaincode if needed
  echo -e "${YELLOW}Deploying NeuraShield chaincode...${NC}"
  bash "${BLOCKCHAIN_SCRIPTS}/deploy-neurashield.sh"
else
  echo -e "${GREEN}✓ Fabric network is already running${NC}"
fi

# Start the server
echo -e "${YELLOW}Starting NeuraShield server...${NC}"
bash "${SERVER_SCRIPTS}/run-server.sh"

echo -e "${GREEN}✓ NeuraShield is now running${NC}"
echo -e "${YELLOW}You can access the application at http://localhost:3000${NC}"
echo -e "${YELLOW}To stop the server, run: kill \$(cat backend/server.pid)${NC}"
echo -e "${YELLOW}To stop the fabric network, run: cd fabric-setup/fabric-samples/test-network && ./network.sh down${NC}" 