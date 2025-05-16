#!/bin/bash

# NeuraShield Test Runner
# This script runs all tests for the NeuraShield platform

# Color definitions for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=======================================${NC}"
echo -e "${YELLOW}       NeuraShield Test Runner        ${NC}"
echo -e "${YELLOW}=======================================${NC}"

# Create directory for test results
RESULTS_DIR="test_results"
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/test_results_$TIMESTAMP.log"

# Track test results
PASSED=0
FAILED=0
TOTAL=0

# Function to run a test and track results
run_test() {
    TEST_NAME=$1
    TEST_CMD=$2
    
    echo -e "\n${YELLOW}Running test: ${TEST_NAME}${NC}"
    echo "--------------------------------------"
    
    # Log test information
    echo "Test: $TEST_NAME" >> $RESULTS_FILE
    echo "Command: $TEST_CMD" >> $RESULTS_FILE
    echo "Start time: $(date)" >> $RESULTS_FILE
    
    # Run the test
    $TEST_CMD
    TEST_RESULT=$?
    
    echo "End time: $(date)" >> $RESULTS_FILE
    echo "Result: $TEST_RESULT" >> $RESULTS_FILE
    echo "--------------------------------------" >> $RESULTS_FILE
    
    # Update counters
    TOTAL=$((TOTAL+1))
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ Test passed!${NC}"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}❌ Test failed!${NC}"
        FAILED=$((FAILED+1))
    fi
    
    echo ""
}

# Install required npm packages if not already installed
if [ ! -d "../node_modules/supertest" ]; then
    echo -e "${YELLOW}Installing required npm packages...${NC}"
    cd .. && npm install --save-dev supertest express && cd test
fi

# Run blockchain verification tests
echo -e "\n${YELLOW}=== Running Blockchain Verification Tests ===${NC}"
run_test "Blockchain Verification" "node blockchain_verification_test.js"

# Run GCP monitoring tests
echo -e "\n${YELLOW}=== Running GCP Monitoring Tests ===${NC}"
run_test "GCP Monitoring" "node gcp_monitoring_test.js"

# Run API tests
echo -e "\n${YELLOW}=== Running API Tests ===${NC}"
run_test "API Tests" "./run_api_test.sh"

# Print summary
echo -e "\n${YELLOW}=======================================${NC}"
echo -e "${YELLOW}            Test Summary              ${NC}"
echo -e "${YELLOW}=======================================${NC}"
echo -e "Total tests:  $TOTAL"
echo -e "Passed:      ${GREEN}$PASSED${NC}"
echo -e "Failed:      ${RED}$FAILED${NC}"
echo -e "\nDetailed results written to: $RESULTS_FILE"

# Exit with code 1 if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0 