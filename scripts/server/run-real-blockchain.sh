#!/bin/bash

# Exit on any error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Starting NeuraShield with REAL Blockchain Integration =====${NC}"

# Make sure blockchain-related settings are correctly set
export SKIP_BLOCKCHAIN=false
export BLOCKCHAIN_ENABLED=true
export FABRIC_TLS_ENABLED=true
export FABRIC_VERIFY_HOSTNAME=true

# Check if the fabric network is running by checking for peer0.org1.example.com docker container
if ! docker ps | grep peer0.org1.example.com > /dev/null; then
    echo -e "${RED}Error: Fabric network is not running.${NC}"
    echo -e "${YELLOW}Please start the Fabric network first with:${NC}"
    echo -e "cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network && ./network.sh up -ca -s couchdb"
    exit 1
fi

echo -e "${GREEN}✓ Fabric network is running${NC}"

# Verify TLS certificates
ORG1_CERT="/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
if [ ! -f "$ORG1_CERT" ]; then
    echo -e "${RED}Error: TLS certificates not found. Fabric network might not be set up correctly.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ TLS certificates verified${NC}"

# Kill any existing server process
if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
    echo -e "${YELLOW}Stopping existing server process...${NC}"
    kill $(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
    rm -f /home/jub/Cursor/neurashield/backend/server.pid
    echo -e "${GREEN}✓ Existing server stopped${NC}"
    # Give it a moment to fully stop
    sleep 2
fi

# Create logs directory if it doesn't exist
mkdir -p /home/jub/Cursor/neurashield/logs

# Start the server with real blockchain
echo -e "${YELLOW}Starting server with real blockchain integration...${NC}"
cd /home/jub/Cursor/neurashield/backend
export PORT=3000
export LOG_LEVEL=info
export NODE_ENV=development
export CHANNEL_NAME=neurashield-channel
export CHAINCODE_NAME=neurashield

# Start the server
node src/server.js > ../logs/server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > server.pid
echo -e "${GREEN}✓ Server started with PID ${SERVER_PID}${NC}"

# Wait a moment for the server to initialize
sleep 2

# Check if the server is actually running
if ! ps -p $SERVER_PID > /dev/null; then
    echo -e "${RED}Error: Server failed to start. Check logs at: /home/jub/Cursor/neurashield/logs/server.log${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Server successfully started with real blockchain integration${NC}"
echo -e "${YELLOW}You can access the API at http://localhost:3000/api/health${NC}"
echo -e "${YELLOW}To stop the server, run: kill \$(cat /home/jub/Cursor/neurashield/backend/server.pid)${NC}"

echo -e "${YELLOW}===== NeuraShield Started =====${NC}" 