#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Deploying Chaincode to NeuraShield Network =====${NC}"

# Set environment variables
export FABRIC_CFG_PATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/config/
export CORE_PEER_TLS_ENABLED=true
export PEER0_ORG1_CA=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export PEER0_ORG2_CA=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export ORDERER_CA=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

# Set peer environment variables for Org1
setOrg1Env() {
  export CORE_PEER_LOCALMSPID="Org1MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG1_CA
  export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
  export CORE_PEER_ADDRESS=localhost:7051
}

# Set peer environment variables for Org2
setOrg2Env() {
  export CORE_PEER_LOCALMSPID="Org2MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE=$PEER0_ORG2_CA
  export CORE_PEER_MSPCONFIGPATH=/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
  export CORE_PEER_ADDRESS=localhost:9051
}

# Package the chaincode
packageChaincode() {
  echo -e "${YELLOW}Packaging chaincode...${NC}"
  cd /home/jub/Cursor/neurashield/backend/chaincode
  
  # Create vendor dependencies
  GO111MODULE=on go mod vendor
  
  mkdir -p neurashield-pkg
  cp chaincode.go neurashield-pkg/
  cp -r vendor neurashield-pkg/
  cp go.mod neurashield-pkg/
  cp go.sum neurashield-pkg/
  
  cat > neurashield-pkg/metadata.json << EOF
{
  "type": "golang",
  "label": "neurashield_1.0"
}
EOF
  
  peer lifecycle chaincode package neurashield.tar.gz --path neurashield-pkg --lang golang --label neurashield_1.0
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to package chaincode${NC}"
    exit 1
  fi
  echo -e "${GREEN}Chaincode packaged successfully${NC}"
  cd -
}

# Install chaincode
installChaincode() {
  echo -e "${YELLOW}Installing chaincode on peer0.org1...${NC}"
  setOrg1Env
  peer lifecycle chaincode install /home/jub/Cursor/neurashield/backend/chaincode/neurashield.tar.gz
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install chaincode on peer0.org1${NC}"
    exit 1
  fi
  echo -e "${GREEN}Chaincode installed on peer0.org1${NC}"

  echo -e "${YELLOW}Installing chaincode on peer0.org2...${NC}"
  setOrg2Env
  peer lifecycle chaincode install /home/jub/Cursor/neurashield/backend/chaincode/neurashield.tar.gz
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install chaincode on peer0.org2${NC}"
    exit 1
  fi
  echo -e "${GREEN}Chaincode installed on peer0.org2${NC}"
}

# Query the installed chaincode to get package ID
queryInstalledChaincode() {
  echo -e "${YELLOW}Querying installed chaincode...${NC}"
  setOrg1Env
  peer lifecycle chaincode queryinstalled > /tmp/chaincode_id.txt
  PACKAGE_ID=$(sed -n 's/.*Package ID: \(.*\), Label: neurashield_1.0/\1/p' /tmp/chaincode_id.txt)
  
  if [ -z "$PACKAGE_ID" ]; then
    echo -e "${RED}Failed to get chaincode package ID${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Package ID: ${PACKAGE_ID}${NC}"
}

# Approve chaincode for Org1
approveForOrg1() {
  echo -e "${YELLOW}Approving chaincode for org1...${NC}"
  setOrg1Env
  peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID neurashield-channel --name neurashield --version 1.0 --package-id $PACKAGE_ID \
    --sequence 1 --tls --cafile $ORDERER_CA
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to approve chaincode for org1${NC}"
    exit 1
  fi
  echo -e "${GREEN}Chaincode approved for org1${NC}"
}

# Approve chaincode for Org2
approveForOrg2() {
  echo -e "${YELLOW}Approving chaincode for org2...${NC}"
  setOrg2Env
  peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID neurashield-channel --name neurashield --version 1.0 --package-id $PACKAGE_ID \
    --sequence 1 --tls --cafile $ORDERER_CA
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to approve chaincode for org2${NC}"
    exit 1
  fi
  echo -e "${GREEN}Chaincode approved for org2${NC}"
}

# Check commit readiness
checkCommitReadiness() {
  echo -e "${YELLOW}Checking commit readiness...${NC}"
  setOrg1Env
  peer lifecycle chaincode checkcommitreadiness --channelID neurashield-channel --name neurashield \
    --version 1.0 --sequence 1 --tls --cafile $ORDERER_CA --output json
}

# Commit chaincode
commitChaincode() {
  echo -e "${YELLOW}Committing chaincode...${NC}"
  setOrg1Env
  peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID neurashield-channel --name neurashield --version 1.0 --sequence 1 \
    --tls --cafile $ORDERER_CA --peerAddresses localhost:7051 --tlsRootCertFiles $PEER0_ORG1_CA \
    --peerAddresses localhost:9051 --tlsRootCertFiles $PEER0_ORG2_CA
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to commit chaincode${NC}"
    exit 1
  fi
  echo -e "${GREEN}Chaincode committed${NC}"
}

# Initialize the ledger
initLedger() {
  echo -e "${YELLOW}Initializing the ledger...${NC}"
  setOrg1Env
  peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --tls --cafile $ORDERER_CA -C neurashield-channel -n neurashield \
    --peerAddresses localhost:7051 --tlsRootCertFiles $PEER0_ORG1_CA \
    --peerAddresses localhost:9051 --tlsRootCertFiles $PEER0_ORG2_CA \
    -c '{"function":"InitLedger","Args":[]}' --waitForEvent
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to initialize ledger${NC}"
    exit 1
  fi
  echo -e "${GREEN}Ledger initialized${NC}"
}

# Query chaincode
queryChaincode() {
  echo -e "${YELLOW}Querying chaincode...${NC}"
  setOrg1Env
  peer chaincode query -C neurashield-channel -n neurashield -c '{"function":"QueryAllEvents","Args":[]}'
}

# Execute deployment steps
packageChaincode
installChaincode
queryInstalledChaincode
approveForOrg1
approveForOrg2
checkCommitReadiness
commitChaincode
initLedger
queryChaincode

echo -e "${GREEN}===== Chaincode Deployment Complete =====${NC}" 