#!/bin/bash
set -e

echo "Starting NeuraShield Blockchain Network..."

# Create directory for channel artifacts
mkdir -p /opt/neurashield/blockchain/network/channel-artifacts

# Check if network is already initialized
if [ ! -f /opt/neurashield/blockchain/network/channel-artifacts/genesis.block ]; then
  echo "Initializing blockchain network..."
  
  # Generate certificates using cryptogen
  cryptogen generate --config=/opt/neurashield/blockchain/network/organizations/cryptogen/crypto-config.yaml --output=/opt/neurashield/blockchain/network/organizations
  
  # Generate genesis block and channel transaction
  configtxgen -profile TwoOrgsOrdererGenesis -channelID system-channel -outputBlock /opt/neurashield/blockchain/network/channel-artifacts/genesis.block
  configtxgen -profile TwoOrgsChannel -outputCreateChannelTx /opt/neurashield/blockchain/network/channel-artifacts/neurashield-channel.tx -channelID neurashield-channel
  
  # Generate anchor peer transactions
  configtxgen -profile TwoOrgsChannel -outputAnchorPeersUpdate /opt/neurashield/blockchain/network/channel-artifacts/Org1MSPanchors.tx -channelID neurashield-channel -asOrg Org1MSP
  configtxgen -profile TwoOrgsChannel -outputAnchorPeersUpdate /opt/neurashield/blockchain/network/channel-artifacts/Org2MSPanchors.tx -channelID neurashield-channel -asOrg Org2MSP
fi

# Start the network using docker-compose
cd /opt/neurashield/blockchain/network
docker-compose -f docker-compose-fabric.yml up -d

# Wait for network to start
sleep 10

# Create and join channel
peer channel create -o orderer.example.com:7050 -c neurashield-channel -f /opt/neurashield/blockchain/network/channel-artifacts/neurashield-channel.tx
peer channel join -b neurashield-channel.block

# Install and instantiate chaincode
peer chaincode install -n neurashield -v 1.0 -p /opt/neurashield/blockchain/network/chaincode/
peer chaincode instantiate -o orderer.example.com:7050 -C neurashield-channel -n neurashield -v 1.0 -c '{"Args":["init"]}'

# Start a simple API server
echo "Starting API server on port 8080..."
while true; do
  echo '{"status":"running"}' | nc -l -p 8080
done 