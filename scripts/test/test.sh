#!/bin/bash

# NeuraShield Test Suite
# This script runs all tests including system, user acceptance, and performance tests

set -e

# Configuration
TEST_DIR="test"
OUTPUT_DIR="output/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
    
    mkdir -p $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR/$TIMESTAMP
    
    # Activate Python environment if exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    print "Test environment setup complete"
}

# Run unit tests
run_unit_tests() {
    print "Running unit tests..."
    
    python -m pytest $TEST_DIR/unit -v --cov=src --cov-report=html:$OUTPUT_DIR/$TIMESTAMP/coverage
}

# Run integration tests
run_integration_tests() {
    print "Running integration tests..."
    
    python -m pytest $TEST_DIR/integration -v
}

# Run system tests
run_system_tests() {
    print "Running system tests..."
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Run system tests
    python -m pytest $TEST_DIR/system -v
    
    # Stop services
    docker-compose down
}

# Run user acceptance tests
run_user_acceptance_tests() {
    print "Running user acceptance tests..."
    
    # Start the system
    docker-compose up -d
    
    # Run UAT
    python -m pytest $TEST_DIR/uat -v
    
    # Stop the system
    docker-compose down
}

# Run performance tests
run_performance_tests() {
    print "Running performance tests..."
    
    # Start the system
    docker-compose up -d
    
    # Run performance tests
    python -m pytest $TEST_DIR/performance -v
    
    # Generate performance report
    python scripts/utils/generate_performance_report.py $OUTPUT_DIR/$TIMESTAMP
    
    # Stop the system
    docker-compose down
}

# Run security tests
run_security_tests() {
    print "Running security tests..."
    
    # Run security audit
    python -m pytest $TEST_DIR/security -v
    
    # Generate security report
    python scripts/utils/generate_security_report.py $OUTPUT_DIR/$TIMESTAMP
}

# Main test function
main() {
    print "Starting NeuraShield test suite..."
    
    setup_test_env
    
    # Run all test suites
    run_unit_tests
    run_integration_tests
    run_system_tests
    run_user_acceptance_tests
    run_performance_tests
    run_security_tests
    
    print "All tests completed. Results are in $OUTPUT_DIR/$TIMESTAMP"
}

# Run main function
main 