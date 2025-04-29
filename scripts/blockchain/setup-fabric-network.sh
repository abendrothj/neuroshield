#!/bin/bash

# Exit on any error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Setting up NeuraShield Fabric Network =====${NC}"

# Navigate to fabric setup directory
echo -e "${YELLOW}Navigating to fabric-setup directory${NC}"
cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network

# Check if the network is up
echo -e "${YELLOW}Checking network status${NC}"
if docker ps | grep peer0.org1.example.com > /dev/null; then
    echo -e "${GREEN}✓ Fabric network is already running${NC}"
else
    echo -e "${YELLOW}Starting Fabric network...${NC}"
    ./network.sh down
    ./network.sh up createChannel -c neurashield-channel -ca -s couchdb
    echo -e "${GREEN}✓ Fabric network started with channel 'neurashield-channel'${NC}"
fi

# Switch to neurashield directory
cd /home/jub/Cursor/neurashield

# Deploy chaincode
echo -e "${YELLOW}Deploying neurashield chaincode...${NC}"
cd /home/jub/Cursor/neurashield
bash /home/jub/Cursor/neurashield/deploy-simple.sh

# Fix TLS issues in .env file
echo -e "${YELLOW}Fixing TLS settings...${NC}"
cd /home/jub/Cursor/neurashield
bash /home/jub/Cursor/neurashield/new-env-settings.sh

# Restart the server
echo -e "${YELLOW}Restarting the server with proper settings...${NC}"
cd /home/jub/Cursor/neurashield
if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
    kill $(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
    rm -f /home/jub/Cursor/neurashield/backend/server.pid
    sleep 2
fi

# Start server
bash /home/jub/Cursor/neurashield/run-server.sh

echo -e "${GREEN}✓ NeuraShield Fabric Network Setup Complete${NC}"
echo -e "${YELLOW}Try testing the integration with:${NC}"
echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"id\":\"test-event-1\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"type\":\"SECURITY_ALERT\",\"details\":{\"source\":\"integration-test\",\"severity\":\"high\",\"description\":\"Test security event\"}}' http://localhost:3000/api/events" 