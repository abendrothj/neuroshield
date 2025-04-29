#!/bin/bash

set -e

# Default values
VERSION=""
CHANNEL_NAME="neurashield-channel"
CHAINCODE_NAME="neurashield"
CHAINCODE_PATH="/home/jub/Cursor/neurashield/backend/chaincode"
NAMESPACE="neurashield"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --channel)
      CHANNEL_NAME="$2"
      shift 2
      ;;
    --chaincode)
      CHAINCODE_NAME="$2"
      shift 2
      ;;
    --path)
      CHAINCODE_PATH="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check if version is provided
if [ -z "$VERSION" ]; then
  echo "Error: Version must be specified with --version flag"
  echo "Usage: $0 --version X.X [--channel channel-name] [--chaincode chaincode-name] [--path chaincode-path] [--namespace k8s-namespace]"
  exit 1
fi

echo "Updating chaincode in Kubernetes environment..."
echo "Chaincode: $CHAINCODE_NAME"
echo "Version: $VERSION"
echo "Channel: $CHANNEL_NAME"
echo "Namespace: $NAMESPACE"

# Create temp directory for chaincode
TEMP_DIR=$(mktemp -d)
echo "Creating temporary directory: $TEMP_DIR"
cp -r $CHAINCODE_PATH/* $TEMP_DIR/

# Create chaincode package
echo "Packaging chaincode..."
kubectl exec -n $NAMESPACE cli -- peer lifecycle chaincode package ${CHAINCODE_NAME}_${VERSION}.tar.gz \
  --path /opt/gopath/src/github.com/chaincode/ \
  --lang golang \
  --label ${CHAINCODE_NAME}_${VERSION}

# Copy chaincode to CLI pod
echo "Copying chaincode to CLI pod..."
kubectl cp $TEMP_DIR/. $NAMESPACE/cli:/opt/gopath/src/github.com/chaincode/

# Install chaincode on Org1
echo "Installing chaincode on Org1..."
kubectl exec -n $NAMESPACE cli -- bash -c "CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp \
  CORE_PEER_ADDRESS=peer0.org1.example.com:7051 \
  CORE_PEER_LOCALMSPID=Org1MSP \
  CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  peer lifecycle chaincode install ${CHAINCODE_NAME}_${VERSION}.tar.gz"

# Install chaincode on Org2
echo "Installing chaincode on Org2..."
kubectl exec -n $NAMESPACE cli -- bash -c "CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp \
  CORE_PEER_ADDRESS=peer0.org2.example.com:7051 \
  CORE_PEER_LOCALMSPID=Org2MSP \
  CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
  peer lifecycle chaincode install ${CHAINCODE_NAME}_${VERSION}.tar.gz"

# Get the package ID
PACKAGE_ID=$(kubectl exec -n $NAMESPACE cli -- bash -c "CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp \
  CORE_PEER_ADDRESS=peer0.org1.example.com:7051 \
  CORE_PEER_LOCALMSPID=Org1MSP \
  CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  peer lifecycle chaincode queryinstalled" | grep ${CHAINCODE_NAME}_${VERSION} | awk '{print $3}' | sed 's/,//')
echo "Package ID: $PACKAGE_ID"

# Approve for Org1
echo "Approving chaincode for Org1..."
kubectl exec -n $NAMESPACE cli -- bash -c "CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp \
  CORE_PEER_ADDRESS=peer0.org1.example.com:7051 \
  CORE_PEER_LOCALMSPID=Org1MSP \
  CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  peer lifecycle chaincode approveformyorg -o orderer.example.com:7050 \
  --channelID $CHANNEL_NAME --name $CHAINCODE_NAME --version $VERSION \
  --package-id $PACKAGE_ID --sequence $VERSION \
  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"

# Approve for Org2
echo "Approving chaincode for Org2..."
kubectl exec -n $NAMESPACE cli -- bash -c "CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp \
  CORE_PEER_ADDRESS=peer0.org2.example.com:7051 \
  CORE_PEER_LOCALMSPID=Org2MSP \
  CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
  peer lifecycle chaincode approveformyorg -o orderer.example.com:7050 \
  --channelID $CHANNEL_NAME --name $CHAINCODE_NAME --version $VERSION \
  --package-id $PACKAGE_ID --sequence $VERSION \
  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"

# Commit chaincode definition
echo "Committing chaincode definition..."
kubectl exec -n $NAMESPACE cli -- bash -c "peer lifecycle chaincode commit -o orderer.example.com:7050 \
  --channelID $CHANNEL_NAME --name $CHAINCODE_NAME --version $VERSION \
  --sequence $VERSION \
  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  --peerAddresses peer0.org1.example.com:7051 \
  --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  --peerAddresses peer0.org2.example.com:7051 \
  --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

# Clean up
echo "Cleaning up temporary directory..."
rm -rf $TEMP_DIR

echo "Chaincode $CHAINCODE_NAME updated to version $VERSION successfully!" 