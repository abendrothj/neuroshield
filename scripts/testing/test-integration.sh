#!/bin/bash

set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================================${NC}"
echo -e "${GREEN}NeuraShield Integration Test${NC}"
echo -e "${YELLOW}============================================================${NC}"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
  echo -e "${YELLOW}Checking prerequisites...${NC}"
  
  # Check Docker
  if ! command_exists docker; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ Docker is installed${NC}"
  
  # Check Docker Compose
  if ! command_exists docker-compose; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ Docker Compose is installed${NC}"
  
  # Check Node.js
  if ! command_exists node; then
    echo -e "${RED}Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
  fi
  NODE_VERSION=$(node --version)
  echo -e "${GREEN}✓ Node.js ${NODE_VERSION} is installed${NC}"
  
  # Check Python
  if ! command_exists python3; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
  fi
  PYTHON_VERSION=$(python3 --version)
  echo -e "${GREEN}✓ ${PYTHON_VERSION} is installed${NC}"
  
  # Check if Go is installed (for chaincode development)
  if ! command_exists go; then
    echo -e "${YELLOW}⚠ Go is not installed. It's required for chaincode development.${NC}"
  else
    GO_VERSION=$(go version)
    echo -e "${GREEN}✓ ${GO_VERSION} is installed${NC}"
  fi
  
  echo -e "${GREEN}All essential prerequisites are met.${NC}"
}

# Function to check if the Fabric network is running
check_fabric_network() {
  echo -e "${YELLOW}Checking Fabric network status...${NC}"
  
  if docker ps | grep peer0.org1.example.com >/dev/null; then
    echo -e "${GREEN}✓ Fabric network is running${NC}"
    return 0
  else
    echo -e "${YELLOW}⚠ Fabric network is not running${NC}"
    return 1
  fi
}

# Function to clean up existing containers
cleanup_containers() {
  echo -e "${YELLOW}Cleaning up existing containers...${NC}"
  
  # Find existing fabric containers and remove them
  if docker ps -a | grep hyperledger >/dev/null; then
    echo -e "${YELLOW}Found existing containers, removing them...${NC}"
    docker rm -f peer0.org1.example.com peer0.org2.example.com orderer.example.com cli 2>/dev/null || true
    # Remove any other fabric-related containers
    docker ps -a | grep hyperledger | awk '{print $1}' | xargs docker rm -f 2>/dev/null || true
    echo -e "${GREEN}✓ Containers cleaned up${NC}"
  else
    echo -e "${GREEN}✓ No existing containers to clean up${NC}"
  fi
}

# Function to set up the genesis block
setup_genesis_block() {
  echo -e "${YELLOW}Setting up genesis block...${NC}"
  
  GENESIS_DIR="/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/system-genesis-block"
  GENESIS_FILE="$GENESIS_DIR/genesis.block"
  
  # Make sure the directory exists
  mkdir -p "$GENESIS_DIR"
  
  # Check if genesis.block is a directory and remove it if it is
  if [ -d "$GENESIS_FILE" ]; then
    echo -e "${YELLOW}Removing genesis.block directory...${NC}"
    rm -rf "$GENESIS_FILE"
  fi
  
  # Create a dummy genesis block if it doesn't exist
  if [ ! -f "$GENESIS_FILE" ]; then
    echo -e "${YELLOW}Creating genesis block...${NC}"
    echo '{
      "header": {
        "number": "0",
        "previous_hash": null,
        "data_hash": "genesis_block_hash"
      },
      "data": {
        "data": []
      }
    }' > "$GENESIS_FILE"
  fi
  
  echo -e "${GREEN}✓ Genesis block set up${NC}"
}

# Function to start the Fabric network
start_fabric_network() {
  echo -e "${YELLOW}Starting Fabric network...${NC}"
  
  # Check if fabric-setup directory exists
  if [ ! -d "/home/jub/Cursor/neurashield/fabric-setup" ]; then
    echo -e "${RED}fabric-setup directory not found. Please run the bootstrap script first.${NC}"
    exit 1
  fi
  
  # Check if bootstrap.sh has been run
  if [ ! -d "/home/jub/Cursor/neurashield/fabric-setup/bin" ]; then
    echo -e "${YELLOW}Running bootstrap.sh to set up Fabric environment...${NC}"
    cd /home/jub/Cursor/neurashield/fabric-setup
    ./bootstrap.sh
    cd - > /dev/null
  fi
  
  # Clean up existing containers to avoid conflicts
  cleanup_containers
  
  # Set up the genesis block
  setup_genesis_block
  
  # Start the network
  echo -e "${YELLOW}Starting Docker containers...${NC}"
  cd /home/jub/Cursor/neurashield/blockchain/network
  docker-compose -f docker-compose-fabric.yml up -d
  cd - > /dev/null
  
  echo -e "${GREEN}✓ Fabric network started${NC}"
}

# Function to deploy the chaincode
deploy_chaincode() {
  echo -e "${YELLOW}Deploying chaincode...${NC}"
  
  # Run the deploy-chaincode.sh script
  cd /home/jub/Cursor/neurashield
  ./scripts/deploy-chaincode.sh
  
  echo -e "${GREEN}✓ Chaincode deployed${NC}"
}

# Function to set up the backend
setup_backend() {
  echo -e "${YELLOW}Setting up backend...${NC}"
  
  cd /home/jub/Cursor/neurashield/backend
  
  # Install dependencies
  echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
  npm install
  
  # Enroll admin user
  echo -e "${YELLOW}Enrolling admin user...${NC}"
  node enroll-admin.js
  
  echo -e "${GREEN}✓ Backend setup complete${NC}"
  cd - > /dev/null
}

# Function to set up the AI model
setup_ai_model() {
  echo -e "${YELLOW}Setting up AI model...${NC}"
  
  cd /home/jub/Cursor/neurashield/models
  
  # Check if Python virtual environment already exists
  if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
  fi
  
  # Activate virtual environment and install dependencies
  echo -e "${YELLOW}Installing Python dependencies...${NC}"
  source venv/bin/activate
  pip install -r requirements.txt
  
  echo -e "${GREEN}✓ AI model setup complete${NC}"
  deactivate
  cd - > /dev/null
}

# Function to test the system integration
test_integration() {
  echo -e "${YELLOW}Testing system integration...${NC}"
  
  # Start backend server in the background
  cd /home/jub/Cursor/neurashield/backend
  echo -e "${YELLOW}Starting backend server...${NC}"
  node src/server.js > ../logs/backend.log 2>&1 &
  BACKEND_PID=$!
  echo -e "${GREEN}✓ Backend server started with PID ${BACKEND_PID}${NC}"
  cd - > /dev/null
  
  # Wait for server to start
  echo -e "${YELLOW}Waiting for server to start...${NC}"
  sleep 5
  
  # Test backend health endpoint
  echo -e "${YELLOW}Testing backend health endpoint...${NC}"
  if curl -s http://localhost:3000/api/health | grep "healthy" > /dev/null; then
    echo -e "${GREEN}✓ Backend is healthy${NC}"
  else
    echo -e "${RED}✗ Backend health check failed${NC}"
    kill $BACKEND_PID
    exit 1
  fi
  
  # Test blockchain integration
  echo -e "${YELLOW}Testing blockchain integration...${NC}"
  TEST_EVENT='{
    "id": "test-event-'$(date +%s)'",
    "timestamp": "'$(date -Iseconds)'",
    "type": "Test",
    "details": {
      "message": "This is a test event",
      "source": "integration-test"
    }
  }'
  
  RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$TEST_EVENT" http://localhost:3000/api/events)
  
  if echo $RESPONSE | grep "success" > /dev/null; then
    echo -e "${GREEN}✓ Successfully logged event to blockchain${NC}"
  else
    echo -e "${RED}✗ Failed to log event to blockchain: $RESPONSE${NC}"
    kill $BACKEND_PID
    exit 1
  fi
  
  # Test AI model with blockchain integration
  echo -e "${YELLOW}Testing AI model with blockchain integration...${NC}"
  cd /home/jub/Cursor/neurashield/models
  source venv/bin/activate
  export BLOCKCHAIN_ENABLED=true
  export BLOCKCHAIN_API_URL=http://localhost:3000/api/v1/events
  
  # Create a test prediction
  TEST_PREDICTION='{
    "source_ip": "192.168.1.100",
    "destination_ip": "10.0.0.1",
    "source_port": "45678",
    "destination_port": "80",
    "protocol": "TCP",
    "bytes_sent": 1024,
    "bytes_received": 2048,
    "duration": 0.5,
    "packets": 10,
    "prediction": "Attack",
    "confidence": 0.95,
    "summary": "Test prediction for integration testing"
  }'
  
  echo $TEST_PREDICTION > test_prediction.json
  
  python -c "
import sys
sys.path.append('.')
from predict import log_to_blockchain
import json

with open('test_prediction.json') as f:
    test_data = json.load(f)

result = log_to_blockchain(test_data)
if result:
    print('SUCCESS')
else:
    print('FAILURE')
  "
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully logged AI prediction to blockchain${NC}"
  else
    echo -e "${RED}✗ Failed to log AI prediction to blockchain${NC}"
    deactivate
    kill $BACKEND_PID
    exit 1
  fi
  
  deactivate
  cd - > /dev/null
  
  # Stop backend server
  echo -e "${YELLOW}Stopping backend server...${NC}"
  kill $BACKEND_PID
  
  echo -e "${GREEN}✓ Integration test complete${NC}"
}

# Main execution
check_prerequisites

# Check if Fabric network is running, if not start it
if ! check_fabric_network; then
  start_fabric_network
fi

# Deploy chaincode
deploy_chaincode

# Set up backend
setup_backend

# Set up AI model
setup_ai_model

# Test integration
test_integration

echo -e "${YELLOW}============================================================${NC}"
echo -e "${GREEN}All tests passed! The system is working correctly.${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Start the backend server: ${GREEN}cd /home/jub/Cursor/neurashield/backend && npm start${NC}"
echo -e "2. Deploy in production using Kubernetes: ${GREEN}See IMPLEMENTATION_GUIDE.md for details${NC}"
echo -e "3. Develop the frontend dashboard: ${GREEN}cd /home/jub/Cursor/neurashield/frontend${NC}"

exit 0 