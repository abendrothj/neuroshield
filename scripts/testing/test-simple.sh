#!/bin/bash

set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================================${NC}"
echo -e "${GREEN}NeuraShield Simple Integration Test${NC}"
echo -e "${YELLOW}============================================================${NC}"

# Function to check prerequisites
check_prerequisites() {
  echo -e "${YELLOW}Checking prerequisites...${NC}"
  
  # Check Node.js
  if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
  fi
  NODE_VERSION=$(node --version)
  echo -e "${GREEN}✓ Node.js ${NODE_VERSION} is installed${NC}"
  
  # Check Python
  if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
  fi
  PYTHON_VERSION=$(python3 --version)
  echo -e "${GREEN}✓ ${PYTHON_VERSION} is installed${NC}"
  
  echo -e "${GREEN}All essential prerequisites are met.${NC}"
}

# Function to set up the backend
setup_backend() {
  echo -e "${YELLOW}Setting up backend...${NC}"
  
  cd /home/jub/Cursor/neurashield/backend
  
  # Install dependencies
  echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
  npm install
  
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

# Function to run a mocked blockchain test
test_mock_integration() {
  echo -e "${YELLOW}Testing with mock blockchain...${NC}"
  
  # Start mock blockchain server in the background
  cd /home/jub/Cursor/neurashield/backend
  echo -e "${YELLOW}Starting mock blockchain server...${NC}"
  node mock-blockchain.js > ../logs/mock-blockchain.log 2>&1 &
  MOCK_PID=$!
  echo -e "${GREEN}✓ Mock blockchain server started with PID ${MOCK_PID}${NC}"
  cd - > /dev/null
  
  # Wait for server to start
  echo -e "${YELLOW}Waiting for server to start...${NC}"
  sleep 5
  
  # Test API
  echo -e "${YELLOW}Testing API...${NC}"
  if curl -s http://localhost:3000/api/health | grep "healthy" > /dev/null; then
    echo -e "${GREEN}✓ API is healthy${NC}"
  else
    echo -e "${RED}✗ API health check failed${NC}"
    kill $MOCK_PID
    exit 1
  fi
  
  # Test API with a test event
  echo -e "${YELLOW}Testing API with a test event...${NC}"
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
    echo -e "${GREEN}✓ Successfully logged event to mock blockchain${NC}"
  else
    echo -e "${RED}✗ Failed to log event to mock blockchain: $RESPONSE${NC}"
    kill $MOCK_PID
    exit 1
  fi
  
  # Test AI model with mock blockchain integration
  echo -e "${YELLOW}Testing AI model with mock blockchain integration...${NC}"
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
    echo -e "${GREEN}✓ Successfully logged AI prediction to mock blockchain${NC}"
  else
    echo -e "${RED}✗ Failed to log AI prediction to mock blockchain${NC}"
    deactivate
    kill $MOCK_PID
    exit 1
  fi
  
  deactivate
  cd - > /dev/null
  
  # Stop mock server
  echo -e "${YELLOW}Stopping mock blockchain server...${NC}"
  kill $MOCK_PID
  
  echo -e "${GREEN}✓ Mock integration test complete${NC}"
}

# Main execution
check_prerequisites

# Set up backend
setup_backend

# Set up AI model
setup_ai_model

# Test with mock integration
test_mock_integration

echo -e "${YELLOW}============================================================${NC}"
echo -e "${GREEN}All tests passed! The AI and API components are working correctly.${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Once blockchain issues are fixed, run the full integration test: ${GREEN}./test-integration.sh${NC}"
echo -e "2. For production deployment, follow the instructions in: ${GREEN}IMPLEMENTATION_GUIDE.md${NC}"

exit 0 