#!/bin/bash

# Exit on error
set -e

# Chaincode details
CHAINCODE_NAME="neurashield"
CHAINCODE_VERSION="1.0"
CHAINCODE_SEQUENCE="1"
CHAINCODE_PATH="chaincode/neurashield"
CHANNEL_NAME="mychannel"

# Package the chaincode
echo "Packaging chaincode..."
peer lifecycle chaincode package ${CHAINCODE_NAME}.tar.gz \
    --path ${CHAINCODE_PATH} \
    --lang golang \
    --label ${CHAINCODE_NAME}_${CHAINCODE_VERSION}

# Install chaincode on Org1 peer
echo "Installing chaincode on Org1 peer..."
peer lifecycle chaincode install ${CHAINCODE_NAME}.tar.gz

# Install chaincode on Org2 peer
echo "Installing chaincode on Org2 peer..."
export CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=peer0.org2.example.com:9051
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt

peer lifecycle chaincode install ${CHAINCODE_NAME}.tar.gz

# Approve chaincode for Org1
echo "Approving chaincode for Org1..."
export CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=peer0.org1.example.com:7051
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt

peer lifecycle chaincode approveformyorg \
    -o orderer.example.com:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --channelID ${CHANNEL_NAME} \
    --name ${CHAINCODE_NAME} \
    --version ${CHAINCODE_VERSION} \
    --package-id ${CHAINCODE_NAME}_${CHAINCODE_VERSION}:$(peer lifecycle chaincode queryinstalled | grep -o "${CHAINCODE_NAME}_${CHAINCODE_VERSION}:[a-z0-9]*") \
    --sequence ${CHAINCODE_SEQUENCE} \
    --tls \
    --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

# Approve chaincode for Org2
echo "Approving chaincode for Org2..."
export CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=peer0.org2.example.com:9051
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt

peer lifecycle chaincode approveformyorg \
    -o orderer.example.com:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --channelID ${CHANNEL_NAME} \
    --name ${CHAINCODE_NAME} \
    --version ${CHAINCODE_VERSION} \
    --package-id ${CHAINCODE_NAME}_${CHAINCODE_VERSION}:$(peer lifecycle chaincode queryinstalled | grep -o "${CHAINCODE_NAME}_${CHAINCODE_VERSION}:[a-z0-9]*") \
    --sequence ${CHAINCODE_SEQUENCE} \
    --tls \
    --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

# Commit chaincode
echo "Committing chaincode..."
export CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=peer0.org1.example.com:7051
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt

peer lifecycle chaincode commit \
    -o orderer.example.com:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --channelID ${CHANNEL_NAME} \
    --name ${CHAINCODE_NAME} \
    --version ${CHAINCODE_VERSION} \
    --sequence ${CHAINCODE_SEQUENCE} \
    --tls \
    --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
    --peerAddresses peer0.org1.example.com:7051 \
    --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses peer0.org2.example.com:9051 \
    --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt

echo "Chaincode deployment completed successfully!" 