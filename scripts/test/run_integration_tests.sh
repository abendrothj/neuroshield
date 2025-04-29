#!/bin/bash

# NeuraShield Integration Test Runner
# This script runs integration tests with proper infrastructure setup

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Print with color
print() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup test environment
setup_test_env() {
    print "Setting up test environment..."
    
    # Start Docker if not running
    if ! docker info > /dev/null 2>&1; then
        print "Starting Docker..."
        sudo service docker start
        sleep 5
    fi

    # Start Fabric network
    print "Starting Hyperledger Fabric network..."
    cd /home/jub/Cursor/neurashield/blockchain/network
    docker-compose -f docker-compose-fabric.yml up -d
    sleep 10  # Wait for network to stabilize

    # Install and instantiate chaincode
    print "Installing chaincode..."
    cd /home/jub/Cursor/neurashield/scripts/deploy
    ./deploy.sh

    # Activate Python environment
    cd /home/jub/Cursor/neurashield
    source venv/bin/activate

    print "Test environment setup complete"
}

# Run integration tests
run_integration_tests() {
    print "Running integration tests..."
    
    # Run blockchain integration tests
    print "Running blockchain integration tests..."
    python -m pytest test/integration/test_blockchain_integration.py -v

    # Run AI model integration tests
    print "Running AI model integration tests..."
    python -m pytest test/integration/test_ai_model.py -v

    # Run system tests
    print "Running system tests..."
    python -m pytest test/system/test_full_system.py -v
}

# Cleanup
cleanup() {
    print "Cleaning up test environment..."
    
    # Stop Fabric network
    cd /home/jub/Cursor/neurashield/blockchain/network
    docker-compose -f docker-compose-fabric.yml down

    print "Cleanup complete"
}

# Main function
main() {
    print "Starting NeuraShield integration tests..."
    
    setup_test_env
    run_integration_tests
    cleanup
    
    print "All tests completed"
}

# Run main function
main 