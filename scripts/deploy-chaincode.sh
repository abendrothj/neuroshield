#!/bin/bash

# Exit on first error
set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}====== Deploying NeuraShield Chaincode ======${NC}"

# Check if Fabric network is running
if ! docker ps | grep peer0.org1.example.com >/dev/null; then
  echo -e "${RED}Fabric network is not running. Please start it first.${NC}"
  exit 1
fi

echo -e "${GREEN}Fabric network is already running.${NC}"

# Set environment variables
export FABRIC_CFG_PATH=/home/jub/Cursor/neurashield/fabric-setup/config
export PATH=/home/jub/Cursor/neurashield/fabric-setup/bin:$PATH
export CHANNEL_NAME=neurashield-channel

# Package the chaincode
echo -e "${YELLOW}Packaging chaincode...${NC}"
cd /home/jub/Cursor/neurashield/backend/chaincode
GO111MODULE=on go mod vendor
cd /home/jub/Cursor/neurashield
peer lifecycle chaincode package neurashield.tar.gz --path ./backend/chaincode --lang golang --label neurashield_1.0

# Install chaincode on Org1
echo -e "${YELLOW}Installing chaincode on Org1...${NC}"
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051
export CORE_PEER_TLS_CLIENTAUTHREQUIRED=false
export CORE_PEER_TLS_VERIFY=false
peer lifecycle chaincode install neurashield.tar.gz

# Install chaincode on Org2
echo -e "${YELLOW}Installing chaincode on Org2...${NC}"
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051
export CORE_PEER_TLS_CLIENTAUTHREQUIRED=false
export CORE_PEER_TLS_VERIFY=false
peer lifecycle chaincode install neurashield.tar.gz

# Query chaincode package ID
echo -e "${YELLOW}Querying chaincode package ID...${NC}"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

PACKAGE_ID=$(peer lifecycle chaincode queryinstalled | grep neurashield_1.0 | awk -F'[: ]' '{print $3}' | tr -d ',')
echo -e "${GREEN}Package ID: ${PACKAGE_ID}${NC}"

# Approve chaincode for Org1
echo -e "${YELLOW}Approving chaincode for Org1...${NC}"
peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID ${CHANNEL_NAME} --name neurashield --version 1.0 --package-id ${PACKAGE_ID} --sequence 1 --tls --cafile /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

# Approve chaincode for Org2
echo -e "${YELLOW}Approving chaincode for Org2...${NC}"
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051
peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID ${CHANNEL_NAME} --name neurashield --version 1.0 --package-id ${PACKAGE_ID} --sequence 1 --tls --cafile /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

# Check commit readiness
echo -e "${YELLOW}Checking commit readiness...${NC}"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051
peer lifecycle chaincode checkcommitreadiness --channelID ${CHANNEL_NAME} --name neurashield --version 1.0 --sequence 1 --tls --cafile /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --output json

# Commit chaincode definition
echo -e "${YELLOW}Committing chaincode definition...${NC}"
peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID ${CHANNEL_NAME} --name neurashield --version 1.0 --sequence 1 --tls --cafile /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt

# Query committed chaincode to confirm
echo -e "${YELLOW}Querying committed chaincode...${NC}"
peer lifecycle chaincode querycommitted --channelID ${CHANNEL_NAME} --name neurashield --cafile /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

echo -e "${GREEN}✓ NeuraShield chaincode successfully deployed!${NC}"

# Initialize the chaincode
echo -e "${YELLOW}Initializing the chaincode...${NC}"
peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem -C ${CHANNEL_NAME} -n neurashield --peerAddresses localhost:7051 --tlsRootCertFiles /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -c '{"function":"InitLedger","Args":[]}'

echo -e "${GREEN}✓ Chaincode initialization completed${NC}" 