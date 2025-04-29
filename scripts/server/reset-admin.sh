#!/bin/bash

# Exit on any error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Resetting Admin Identity in Wallet =====${NC}"

# Check if the Fabric network is running
if ! docker ps | grep peer0.org1.example.com > /dev/null; then
    echo -e "${RED}Error: Fabric network is not running.${NC}"
    echo -e "${YELLOW}Please start the Fabric network first with:${NC}"
    echo -e "cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network && ./network.sh up -ca -s couchdb"
    exit 1
fi

echo -e "${GREEN}✓ Fabric network is running${NC}"

# Define paths
ADMIN_CERT_PATH="/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/signcerts/cert.pem"
ADMIN_KEY_PATH="/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore/$(ls /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore/)"
WALLET_DIR="/home/jub/Cursor/neurashield/backend/wallet"

# Verify paths exist
if [ ! -f "$ADMIN_CERT_PATH" ]; then
    echo -e "${RED}Error: Admin certificate not found at:${NC}"
    echo -e "$ADMIN_CERT_PATH"
    exit 1
fi

if [ ! -f "$ADMIN_KEY_PATH" ]; then
    echo -e "${RED}Error: Admin private key not found at:${NC}"
    echo -e "$ADMIN_KEY_PATH"
    exit 1
fi

echo -e "${GREEN}✓ Admin MSP files found${NC}"

# Create wallet directory if it doesn't exist
if [ ! -d "$WALLET_DIR" ]; then
    echo -e "${YELLOW}Creating wallet directory...${NC}"
    mkdir -p "$WALLET_DIR"
fi

# Backup existing admin identity if it exists
if [ -f "$WALLET_DIR/admin.id" ]; then
    echo -e "${YELLOW}Backing up existing admin identity...${NC}"
    mv "$WALLET_DIR/admin.id" "$WALLET_DIR/admin.id.bak"
fi

if [ -d "$WALLET_DIR/admin" ]; then
    echo -e "${YELLOW}Backing up existing admin directory...${NC}"
    mv "$WALLET_DIR/admin" "$WALLET_DIR/admin.bak"
fi

echo -e "${YELLOW}Creating new admin identity in wallet...${NC}"

# Create admin.id file with proper format
CERT=$(cat "$ADMIN_CERT_PATH" | sed 's/$/\\n/' | tr -d '\n')
KEY=$(cat "$ADMIN_KEY_PATH" | sed 's/$/\\n/' | tr -d '\n')

cat > "$WALLET_DIR/admin.id" << EOF
{"credentials":{"certificate":"-----BEGIN CERTIFICATE-----\\n${CERT}-----END CERTIFICATE-----\\n","privateKey":"-----BEGIN PRIVATE KEY-----\\n${KEY}-----END PRIVATE KEY-----\\n"},"mspId":"Org1MSP","type":"X.509","version":1}
EOF

echo -e "${GREEN}✓ Admin identity reset successfully${NC}"
echo -e "${YELLOW}Restarting server to apply changes...${NC}"

# Kill any existing server process
if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
    kill $(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
    rm -f /home/jub/Cursor/neurashield/backend/server.pid
    # Give it a moment to fully stop
    sleep 2
fi

# Start the server again
cd /home/jub/Cursor/neurashield
bash /home/jub/Cursor/neurashield/run-server.sh

echo -e "${GREEN}===== Admin Identity Reset Complete =====${NC}"
echo -e "${YELLOW}You can test the blockchain connection now.${NC}" 