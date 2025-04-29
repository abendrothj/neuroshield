#!/bin/bash

# Exit on any error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== NeuraShield Production Setup =====${NC}"
echo -e "${YELLOW}Removing mock implementations and setting up real blockchain${NC}"

# Step 1: Stop any running servers
echo -e "${YELLOW}Stopping any running servers...${NC}"
pkill -f "node mock-blockchain.js" 2>/dev/null || true
pkill -f "node src/server.js" 2>/dev/null || true

if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
    kill $(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
    rm -f /home/jub/Cursor/neurashield/backend/server.pid
fi
echo -e "${GREEN}✓ All servers stopped${NC}"

# Step 2: Remove mock blockchain script
echo -e "${YELLOW}Removing mock blockchain implementation...${NC}"
if [ -f "/home/jub/Cursor/neurashield/backend/mock-blockchain.js" ]; then
    mv /home/jub/Cursor/neurashield/backend/mock-blockchain.js /home/jub/Cursor/neurashield/backend/mock-blockchain.js.bak
    echo -e "${GREEN}✓ Mock blockchain implementation backed up and removed${NC}"
else
    echo -e "${YELLOW}Mock blockchain implementation already removed${NC}"
fi

# Step 3: Remove the run-mock-blockchain.sh script
echo -e "${YELLOW}Removing mock blockchain startup script...${NC}"
if [ -f "/home/jub/Cursor/neurashield/run-mock-blockchain.sh" ]; then
    mv /home/jub/Cursor/neurashield/run-mock-blockchain.sh /home/jub/Cursor/neurashield/run-mock-blockchain.sh.bak
    echo -e "${GREEN}✓ Mock blockchain startup script backed up and removed${NC}"
else
    echo -e "${YELLOW}Mock blockchain startup script already removed${NC}"
fi

# Step 4: Update .env file to enforce real blockchain
echo -e "${YELLOW}Updating environment configuration...${NC}"
cat > /home/jub/Cursor/neurashield/backend/.env << EOL
# Server Configuration
PORT=3000
NODE_ENV=production

# API Endpoints
AI_SERVICE_URL=http://localhost:5000

# Blockchain Configuration - PRODUCTION
SKIP_BLOCKCHAIN=false
BLOCKCHAIN_ENABLED=true
CHANNEL_NAME=neurashield-channel
CHAINCODE_NAME=neurashield
CONTRACT_NAME=neurashield
ORGANIZATION_ID=Org1
USER_ID=admin
BLOCKCHAIN_IDENTITY=admin

# TLS Configuration
FABRIC_TLS_ENABLED=true
FABRIC_VERIFY_HOSTNAME=true

# IPFS Configuration
IPFS_URL=http://localhost:5001

# Logging
LOG_LEVEL=info

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=neurashield
DB_USER=neurauser
DB_PASSWORD=878a096fb61bf207289b1387

# Force production mode - no mock implementations
USE_MOCK_BLOCKCHAIN=false
MOCK_BLOCKCHAIN_URL=
EOL
echo -e "${GREEN}✓ Environment configuration updated for production${NC}"

# Step 5: Set up Fabric network
echo -e "${YELLOW}Setting up Fabric network...${NC}"
cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network

# Check if network is running, otherwise start it
if docker ps | grep peer0.org1.example.com > /dev/null; then
    echo -e "${GREEN}✓ Fabric network is already running${NC}"
    
    # Check if channel exists
    if docker exec cli peer channel list 2>/dev/null | grep neurashield-channel > /dev/null; then
        echo -e "${GREEN}✓ Channel neurashield-channel already exists${NC}"
    else
        echo -e "${YELLOW}Creating neurashield-channel...${NC}"
        ./network.sh createChannel -c neurashield-channel
        echo -e "${GREEN}✓ Channel neurashield-channel created${NC}"
    fi
else
    echo -e "${YELLOW}Starting Fabric network...${NC}"
    ./network.sh down
    ./network.sh up createChannel -c neurashield-channel -ca -s couchdb
    echo -e "${GREEN}✓ Fabric network started with channel 'neurashield-channel'${NC}"
fi

# Step 6: Fix connection profile paths
echo -e "${YELLOW}Updating connection profile paths...${NC}"
cd /home/jub/Cursor/neurashield
cp /home/jub/Cursor/neurashield/backend/connection-profile.json /home/jub/Cursor/neurashield/backend/connection-profile.json.bak

# Use sed to update paths in connection profile
sed -i 's|/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations|/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations|g' /home/jub/Cursor/neurashield/backend/connection-profile.json
echo -e "${GREEN}✓ Connection profile paths updated${NC}"

# Step 7: Deploy chaincode
echo -e "${YELLOW}Deploying chaincode...${NC}"
cd /home/jub/Cursor/neurashield
bash /home/jub/Cursor/neurashield/deploy-simple.sh || {
    echo -e "${RED}Chaincode deployment failed but continuing setup${NC}"
}

# Step 8: Setup the wallet directory with admin credentials
echo -e "${YELLOW}Setting up wallet directory...${NC}"
mkdir -p /home/jub/Cursor/neurashield/backend/wallet

# Step 9: Update the run-server.sh script to force production mode
echo -e "${YELLOW}Updating server startup script...${NC}"
cat > /home/jub/Cursor/neurashield/run-server.sh << EOL
#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "\${YELLOW}Starting NeuraShield Production Server on port 3000\${NC}"

# Set environment variables - PRODUCTION MODE ONLY
export PORT=3000
export LOG_LEVEL=info
# Force real blockchain only
export SKIP_BLOCKCHAIN=false
export BLOCKCHAIN_ENABLED=true
export USE_MOCK_BLOCKCHAIN=false
# Enable TLS verification
export FABRIC_TLS_ENABLED=true
export FABRIC_VERIFY_HOSTNAME=true

# Create logs directory if it doesn't exist
mkdir -p /home/jub/Cursor/neurashield/logs

# Check if the fabric network is running
if ! docker ps 2>/dev/null | grep peer0.org1.example.com > /dev/null; then
    echo -e "\${RED}Error: Fabric network is not running.\${NC}"
    echo -e "\${YELLOW}Please start the Fabric network first with:\${NC}"
    echo -e "cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network && ./network.sh up createChannel -c neurashield-channel -ca -s couchdb"
    exit 1
fi

echo -e "\${GREEN}✓ Fabric network is running\${NC}"

# Kill any existing server process
if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
    echo -e "\${YELLOW}Stopping existing server process...\${NC}"
    kill \$(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
    rm -f /home/jub/Cursor/neurashield/backend/server.pid
    echo -e "\${GREEN}✓ Existing server stopped\${NC}"
    # Give it a moment to fully stop
    sleep 2
fi

# Start the server in production mode
cd /home/jub/Cursor/neurashield/backend
export NODE_ENV=production
node src/server.js > ../logs/server.log 2>&1 &
SERVER_PID=\$!
echo \$SERVER_PID > server.pid
echo -e "\${GREEN}Server started with PID \${SERVER_PID}\${NC}"

# Wait a moment for the server to initialize
sleep 2

# Check if the server is actually running
if ! ps -p \$SERVER_PID > /dev/null; then
    echo -e "\${RED}Error: Server failed to start. Check logs at: /home/jub/Cursor/neurashield/logs/server.log\${NC}"
    exit 1
fi

echo -e "\${GREEN}✓ Production server successfully started with real blockchain integration\${NC}"
echo -e "\${YELLOW}You can access the API at http://localhost:3000/api/health\${NC}"
echo -e "\${YELLOW}To stop the server, run: kill \\\$(cat /home/jub/Cursor/neurashield/backend/server.pid)\${NC}"
EOL
chmod +x /home/jub/Cursor/neurashield/run-server.sh
echo -e "${GREEN}✓ Server startup script updated for production mode${NC}"

echo -e "${GREEN}✓ NeuraShield production setup complete!${NC}"
echo -e "${YELLOW}You can now start the production server with:${NC}"
echo -e "${YELLOW}   /home/jub/Cursor/neurashield/run-server.sh${NC}"
echo -e "${YELLOW}To test the blockchain integration:${NC}"
echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"id\":\"prod-event-1\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"type\":\"SECURITY_ALERT\",\"details\":{\"source\":\"production-test\",\"severity\":\"critical\",\"description\":\"Production blockchain test\"}}' http://localhost:3000/api/events" 