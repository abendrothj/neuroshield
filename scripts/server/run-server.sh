#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting NeuraShield Server on port 3000${NC}"

# Set environment variables - PERMANENT BLOCKCHAIN FIXES
export PORT=3000
export LOG_LEVEL=info
# Force real blockchain by default
export SKIP_BLOCKCHAIN=false
export BLOCKCHAIN_ENABLED=true
# Enable TLS verification
export FABRIC_TLS_ENABLED=true
export FABRIC_VERIFY_HOSTNAME=true

# Create logs directory if it doesn't exist
mkdir -p /home/jub/Cursor/neurashield/logs

# Check if the fabric network is running (can be skipped with SKIP_CHECK=true)
if [[ "$SKIP_CHECK" != "true" ]]; then
    if ! docker ps 2>/dev/null | grep peer0.org1.example.com > /dev/null; then
        echo -e "${YELLOW}Warning: Fabric network might not be running.${NC}"
        echo -e "${YELLOW}If this is intentional, ignore this message.${NC}"
        echo -e "${YELLOW}Otherwise, start the Fabric network with:${NC}"
        echo -e "cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network && ./network.sh up -ca -s couchdb"
        echo -e "${YELLOW}To bypass this check, run with SKIP_CHECK=true ${NC}"
        
        # Give user a chance to abort
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Aborted.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Fabric network is running${NC}"
    fi
fi

# Kill any existing server process
if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
    echo -e "${YELLOW}Stopping existing server process...${NC}"
    kill $(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
    rm -f /home/jub/Cursor/neurashield/backend/server.pid
    echo -e "${GREEN}✓ Existing server stopped${NC}"
    # Give it a moment to fully stop
    sleep 2
fi

# Start the server
cd /home/jub/Cursor/neurashield/backend
node src/server.js > ../logs/server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > server.pid
echo -e "${GREEN}Server started with PID ${SERVER_PID}${NC}"

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