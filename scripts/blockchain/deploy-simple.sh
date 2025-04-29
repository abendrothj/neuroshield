#!/bin/bash

# Exit on any error
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Deploying NeuraShield Chaincode using CLI container =====${NC}"

# Copy the chaincode file to CLI container
docker cp /home/jub/Cursor/neurashield/backend/chaincode/chaincode.go cli:/opt/gopath/src/github.com/chaincode/chaincode.go
docker cp /home/jub/Cursor/neurashield/backend/chaincode/go.mod cli:/opt/gopath/src/github.com/chaincode/go.mod
docker cp /home/jub/Cursor/neurashield/backend/chaincode/go.sum cli:/opt/gopath/src/github.com/chaincode/go.sum

# Execute commands in the CLI container
echo -e "${YELLOW}Creating and joining channel${NC}"
docker exec cli bash -c "
  cd /opt/gopath/src/github.com/hyperledger/fabric/peer && 
  peer channel create -o orderer.example.com:7050 -c neurashield-channel -f /opt/gopath/src/github.com/hyperledger/fabric/peer/channel-artifacts/channel.tx --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem || true
"

docker exec cli bash -c "
  cd /opt/gopath/src/github.com/hyperledger/fabric/peer && 
  peer channel join -b neurashield-channel.block || true
"

echo -e "${YELLOW}Packaging and installing chaincode${NC}"
docker exec cli bash -c "
  cd /opt/gopath/src/github.com/chaincode && 
  peer lifecycle chaincode package neurashield.tar.gz --path . --lang golang --label neurashield_1.0
"

docker exec cli bash -c "
  cd /opt/gopath/src/github.com/chaincode && 
  peer lifecycle chaincode install neurashield.tar.gz
"

echo -e "${YELLOW}Getting chaincode package ID${NC}"
docker exec cli bash -c "
  peer lifecycle chaincode queryinstalled
" > /tmp/queryinstalled.txt

PACKAGE_ID=$(grep -o 'Package ID: neurashield_1.0:[^,]*' /tmp/queryinstalled.txt | sed 's/Package ID: //')
echo -e "${GREEN}Package ID: ${PACKAGE_ID}${NC}"

echo -e "${YELLOW}Approving chaincode${NC}"
docker exec cli bash -c "
  peer lifecycle chaincode approveformyorg -o orderer.example.com:7050 --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --channelID neurashield-channel --name neurashield --version 1.0 --sequence 1 --package-id ${PACKAGE_ID}
"

echo -e "${YELLOW}Committing chaincode${NC}"
docker exec cli bash -c "
  peer lifecycle chaincode commit -o orderer.example.com:7050 --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --channelID neurashield-channel --name neurashield --version 1.0 --sequence 1
"

echo -e "${YELLOW}Initializing chaincode${NC}"
docker exec cli bash -c "
  peer chaincode invoke -o orderer.example.com:7050 --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem -C neurashield-channel -n neurashield -c '{\"function\":\"InitLedger\",\"Args\":[]}' --waitForEvent
"

echo -e "${GREEN}✓ Chaincode successfully deployed! Let's test a query:${NC}"
docker exec cli bash -c "
  peer chaincode query -C neurashield-channel -n neurashield -c '{\"function\":\"QueryAllEvents\",\"Args\":[]}'
"

echo -e "${YELLOW}===== Deployment Complete =====${NC}" 