#!/bin/bash

# Exit on any error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Permanently Fixing NeuraShield Blockchain Configuration =====${NC}"

# Create a new .env file with proper settings
cat > /home/jub/Cursor/neurashield/backend/.env << EOL
# Server Configuration
PORT=3000
NODE_ENV=development

# API Endpoints
AI_SERVICE_URL=http://localhost:5000

# Blockchain Configuration
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
EOL

echo -e "${GREEN}✓ Permanently updated .env file with correct blockchain settings${NC}"

# Check for any mock-blockchain script in the run scripts
if grep -q "mock-blockchain" /home/jub/Cursor/neurashield/run-server.sh; then
    echo -e "${YELLOW}Removing mock-blockchain references from run-server.sh${NC}"
    sed -i '/mock-blockchain/d' /home/jub/Cursor/neurashield/run-server.sh
    echo -e "${GREEN}✓ Removed mock-blockchain references${NC}"
fi

# Check for any test scripts that might be using mock blockchain
for script in /home/jub/Cursor/neurashield/*.sh; do
    if grep -q "mock-blockchain.js" "$script" && [[ "$script" != *"test"* ]]; then
        echo -e "${YELLOW}Found mock-blockchain reference in $script${NC}"
        echo -e "${YELLOW}You may want to review this script${NC}"
    fi
done

echo -e "${GREEN}✓ Configuration permanently updated${NC}"
echo -e "${YELLOW}Now you can use ./run-server.sh without needing to fix TLS issues${NC}"

echo -e "${YELLOW}===== Permanent Fix Complete =====${NC}" 