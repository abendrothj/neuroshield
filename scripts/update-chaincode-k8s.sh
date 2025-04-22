#!/bin/bash

# This script packages and installs a new version of the chaincode
# in the Kubernetes deployment

# Exit on error
set -e

# Default values
CHAINCODE_NAME="neurashield"
CHAINCODE_VERSION="1.1"  # Increment for each update
CHAINCODE_PATH="/home/jub/Cursor/neurashield/backend/chaincode"
CHANNEL_NAME="neurashield-channel"
SEQUENCE=$(expr $(peer lifecycle chaincode querycommitted -C $CHANNEL_NAME -n $CHAINCODE_NAME --output json | jq -r '.sequence') + 1)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      CHAINCODE_NAME="$2"
      shift 2
      ;;
    --version)
      CHAINCODE_VERSION="$2"
      shift 2
      ;;
    --path)
      CHAINCODE_PATH="$2"
      shift 2
      ;;
    --channel)
      CHANNEL_NAME="$2"
      shift 2
      ;;
    --sequence)
      SEQUENCE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Updating chaincode in Kubernetes environment"
echo "Name: $CHAINCODE_NAME"
echo "Version: $CHAINCODE_VERSION"
echo "Path: $CHAINCODE_PATH"
echo "Channel: $CHANNEL_NAME"
echo "Sequence: $SEQUENCE"

# Connect to the CLI pod in Kubernetes
CLI_POD=$(kubectl get pods -l component=cli -o jsonpath='{.items[0].metadata.name}')

if [ -z "$CLI_POD" ]; then
  echo "No CLI pod found. Deploying a temporary CLI pod..."
  kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: fabric-cli
  labels:
    component: cli
spec:
  containers:
  - name: cli
    image: hyperledger/fabric-tools:2.5.0
    command: ["sleep", "3600"]
    env:
    - name: GOPATH
      value: /opt/gopath
    - name: FABRIC_LOGGING_SPEC
      value: INFO
    - name: CORE_PEER_ID
      value: cli
    - name: CORE_PEER_ADDRESS
      value: fabric-peer:7051
    - name: CORE_PEER_LOCALMSPID
      value: Org1MSP
    - name: CORE_PEER_TLS_ENABLED
      value: "true"
    - name: CORE_PEER_TLS_CERT_FILE
      value: /etc/hyperledger/fabric/tls/server.crt
    - name: CORE_PEER_TLS_KEY_FILE
      value: /etc/hyperledger/fabric/tls/server.key
    - name: CORE_PEER_TLS_ROOTCERT_FILE
      value: /etc/hyperledger/fabric/tls/ca.crt
    - name: CORE_PEER_MSPCONFIGPATH
      value: /etc/hyperledger/fabric/msp
    volumeMounts:
    - name: peer-config
      mountPath: /etc/hyperledger/fabric
  volumes:
  - name: peer-config
    configMap:
      name: fabric-peer-config
EOF
  
  echo "Waiting for CLI pod to be ready..."
  kubectl wait --for=condition=ready pod/fabric-cli --timeout=60s
  CLI_POD="fabric-cli"
fi

# Copy chaincode to the pod
echo "Copying chaincode to the pod..."
kubectl cp "$CHAINCODE_PATH" "$CLI_POD:/opt/gopath/src/github.com/chaincode"

# Package and install the chaincode
echo "Packaging and installing chaincode..."
kubectl exec "$CLI_POD" -- bash -c "
cd /opt/gopath/src/github.com &&
peer lifecycle chaincode package ${CHAINCODE_NAME}.tar.gz \
  --path ./chaincode \
  --lang golang \
  --label ${CHAINCODE_NAME}_${CHAINCODE_VERSION} &&
peer lifecycle chaincode install ${CHAINCODE_NAME}.tar.gz
"

# Get the package ID
PACKAGE_ID=$(kubectl exec "$CLI_POD" -- peer lifecycle chaincode queryinstalled | grep "${CHAINCODE_NAME}_${CHAINCODE_VERSION}" | awk -F 'ID: ' '{print $2}' | awk -F ',' '{print $1}')
echo "Package ID: $PACKAGE_ID"

# Approve chaincode for Org1
echo "Approving chaincode for Org1..."
kubectl exec "$CLI_POD" -- peer lifecycle chaincode approveformyorg \
  -o fabric-orderer:7050 \
  --channelID "$CHANNEL_NAME" \
  --name "$CHAINCODE_NAME" \
  --version "$CHAINCODE_VERSION" \
  --package-id "$PACKAGE_ID" \
  --sequence "$SEQUENCE" \
  --tls \
  --cafile /etc/hyperledger/fabric/tls/ca.crt

# Commit the chaincode definition
echo "Committing chaincode definition..."
kubectl exec "$CLI_POD" -- peer lifecycle chaincode commit \
  -o fabric-orderer:7050 \
  --channelID "$CHANNEL_NAME" \
  --name "$CHAINCODE_NAME" \
  --version "$CHAINCODE_VERSION" \
  --sequence "$SEQUENCE" \
  --tls \
  --cafile /etc/hyperledger/fabric/tls/ca.crt \
  --peerAddresses fabric-peer:7051 \
  --tlsRootCertFiles /etc/hyperledger/fabric/tls/ca.crt

echo "Chaincode update completed successfully!"

# Clean up temporary CLI pod if we created one
if [ "$CLI_POD" == "fabric-cli" ]; then
  echo "Cleaning up temporary CLI pod..."
  kubectl delete pod fabric-cli
fi 