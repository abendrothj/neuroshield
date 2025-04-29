#!/bin/bash

# Exit on any error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Deploying NeuraShield Chaincode =====${NC}"

# Create a temporary directory for the chaincode to avoid redeclaration issues
TEMP_DIR=$(mktemp -d)
echo -e "${YELLOW}Using temporary directory: ${TEMP_DIR}${NC}"

# Copy the chaincode file to the temporary directory
cp /home/jub/Cursor/neurashield/backend/chaincode/chaincode.go $TEMP_DIR/chaincode.go

# Create go.mod in the temp directory
cd $TEMP_DIR
cat > go.mod << EOL
module github.com/neurashield/chaincode

go 1.14

require github.com/hyperledger/fabric-contract-api-go v1.1.0
EOL

# Initialize Go modules
echo -e "${YELLOW}Initializing Go modules...${NC}"
go mod tidy

echo -e "${YELLOW}Setting up fabric environment...${NC}"
export FABRIC_CFG_PATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/config/
export CORE_PEER_TLS_ENABLED=true
export PEER0_ORG1_CA=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export PEER0_ORG2_CA=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export ORDERER_CA=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

# Set peer environment variables for Org1
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG1_CA
export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

# Query to check if chaincode is already installed
echo -e "${YELLOW}Checking if chaincode is already installed...${NC}"
peer lifecycle chaincode queryinstalled > /tmp/queryinstalled.txt
EXISTING_PACKAGE_ID=$(grep -o 'Package ID: neurashield_1.0:[^,]*' /tmp/queryinstalled.txt | sed 's/Package ID: //' | head -1)

if [ -z "$EXISTING_PACKAGE_ID" ]; then
  echo -e "${YELLOW}Packaging and installing chaincode...${NC}"
  cd /home/jub/Cursor/neurashield
  peer lifecycle chaincode package neurashield.tar.gz --path $TEMP_DIR --lang golang --label neurashield_1.0

  echo -e "${YELLOW}Installing chaincode on peer0.org1...${NC}"
  peer lifecycle chaincode install neurashield.tar.gz

  # Set peer environment variables for Org2
  export CORE_PEER_LOCALMSPID="Org2MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG2_CA
  export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
  export CORE_PEER_ADDRESS=localhost:9051

  echo -e "${YELLOW}Installing chaincode on peer0.org2...${NC}"
  peer lifecycle chaincode install neurashield.tar.gz

  # Query the installed chaincode to get package ID
  echo -e "${YELLOW}Querying installed chaincode...${NC}"
  export CORE_PEER_LOCALMSPID="Org1MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG1_CA
  export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
  export CORE_PEER_ADDRESS=localhost:7051

  peer lifecycle chaincode queryinstalled > /tmp/chaincode_id.txt
  PACKAGE_ID=$(sed -n 's/.*Package ID: \(.*\), Label: neurashield_1.0/\1/p' /tmp/chaincode_id.txt)

  if [ -z "$PACKAGE_ID" ]; then
    echo -e "${RED}Failed to get chaincode package ID${NC}"
    exit 1
  fi

  echo -e "${GREEN}Package ID: ${PACKAGE_ID}${NC}"
else
  echo -e "${GREEN}Chaincode already installed with Package ID: ${EXISTING_PACKAGE_ID}${NC}"
  PACKAGE_ID=$EXISTING_PACKAGE_ID
fi

# Check if chaincode is already approved and committed
echo -e "${YELLOW}Checking if chaincode is already committed...${NC}"
peer lifecycle chaincode querycommitted --channelID neurashield-channel --name neurashield 2>/dev/null || COMMIT_ERROR=$?

if [ ! -z "$COMMIT_ERROR" ]; then
  echo -e "${YELLOW}Approving chaincode for org1...${NC}"
  peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID neurashield-channel --name neurashield --version 1.0 --package-id $PACKAGE_ID \
    --sequence 1 --tls --cafile $ORDERER_CA

  echo -e "${YELLOW}Approving chaincode for org2...${NC}"
  export CORE_PEER_LOCALMSPID="Org2MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG2_CA
  export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
  export CORE_PEER_ADDRESS=localhost:9051

  peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID neurashield-channel --name neurashield --version 1.0 --package-id $PACKAGE_ID \
    --sequence 1 --tls --cafile $ORDERER_CA

  echo -e "${YELLOW}Committing chaincode...${NC}"
  export CORE_PEER_LOCALMSPID="Org1MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG1_CA
  export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
  export CORE_PEER_ADDRESS=localhost:7051

  peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID neurashield-channel --name neurashield --version 1.0 --sequence 1 \
    --tls --cafile $ORDERER_CA --peerAddresses localhost:7051 --tlsRootCertFiles $PEER0_ORG1_CA \
    --peerAddresses localhost:9051 --tlsRootCertFiles $PEER0_ORG2_CA

  echo -e "${YELLOW}Initializing the ledger...${NC}"
  peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --tls --cafile $ORDERER_CA -C neurashield-channel -n neurashield \
    --peerAddresses localhost:7051 --tlsRootCertFiles $PEER0_ORG1_CA \
    --peerAddresses localhost:9051 --tlsRootCertFiles $PEER0_ORG2_CA \
    -c '{"function":"InitLedger","Args":[]}' --waitForEvent
else
  echo -e "${GREEN}Chaincode already approved and committed${NC}"
fi

echo -e "${YELLOW}Querying events...${NC}"
peer chaincode query -C neurashield-channel -n neurashield -c '{"function":"QueryAllEvents","Args":[]}'

# Update environment config
echo -e "${YELLOW}Updating environment settings...${NC}"
# Create or update .env file
if [ -f "/home/jub/Cursor/neurashield/.env" ]; then
  sed -i 's/^CHAINCODE_NAME=.*/CHAINCODE_NAME=neurashield/' /home/jub/Cursor/neurashield/.env
else
  echo "CHAINCODE_NAME=neurashield" > /home/jub/Cursor/neurashield/.env
fi
echo -e "${GREEN}Environment settings updated${NC}"

# Clean up the temporary directory
rm -rf $TEMP_DIR

echo -e "${GREEN}===== NeuraShield Chaincode Deployment Complete =====${NC}" 