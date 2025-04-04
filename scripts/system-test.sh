#!/bin/bash

# NeuraShield System Testing Script
# This script conducts comprehensive validation of the entire NeuraShield platform

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0
TOTAL=0

# Default variables
API_URL="http://localhost:3001"
TESTS_DIR="./tests"
VERBOSE=false
REPORT_FILE="system-test-report-$(date +%Y%m%d_%H%M%S).log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --report)
      REPORT_FILE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: system-test.sh [options]"
      echo "Options:"
      echo "  --api-url URL          Specify API URL (default: http://localhost:3001)"
      echo "  --verbose              Show detailed test output"
      echo "  --report FILE          Specify report file name"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Initialize report file
echo "NeuraShield System Test Report - $(date)" > "$REPORT_FILE"
echo "=======================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Helper function to run a test and log results
run_test() {
  local test_name="$1"
  local test_cmd="$2"
  local test_desc="$3"
  
  echo -e "${YELLOW}Running test: ${test_name}${NC}"
  echo "Test: $test_name - $test_desc" >> "$REPORT_FILE"
  
  TOTAL=$((TOTAL+1))
  
  # Run the test command and capture output
  local output
  local exit_code
  
  output=$(eval "$test_cmd" 2>&1) || exit_code=$?
  
  if [[ -z "$exit_code" ]]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    PASSED=$((PASSED+1))
    echo "Status: PASSED" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ FAILED${NC}"
    FAILED=$((FAILED+1))
    echo "Status: FAILED" >> "$REPORT_FILE"
  fi
  
  if [[ "$VERBOSE" == "true" || "$exit_code" ]]; then
    echo -e "${BLUE}Output:${NC}"
    echo "$output"
    echo "Output:" >> "$REPORT_FILE"
    echo "$output" >> "$REPORT_FILE"
  fi
  
  echo "" >> "$REPORT_FILE"
}

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}NeuraShield Comprehensive System Test${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# 1. API Health Check
run_test "API Health Check" \
  "curl -s -f ${API_URL}/health | grep -q 'status.*ok'" \
  "Verify API is accessible and healthy"

# 2. Authentication Tests
run_test "User Authentication" \
  "curl -s -f -X POST ${API_URL}/auth/login -H 'Content-Type: application/json' -d '{\"username\":\"test_user\",\"password\":\"test_password\"}' | grep -q 'token'" \
  "Verify user authentication works correctly"

# 3. AI Model Tests
run_test "AI Model Status" \
  "curl -s -f ${API_URL}/ai/status | grep -q 'running'" \
  "Verify AI model service is running"

run_test "AI Model Prediction" \
  "curl -s -f -X POST ${API_URL}/ai/predict -H 'Content-Type: application/json' -d '{\"input\":\"test data\"}' | grep -q 'prediction'" \
  "Verify AI model can make predictions"

# 4. Blockchain Tests
run_test "Blockchain Connection" \
  "curl -s -f ${API_URL}/blockchain/status | grep -q 'connected'" \
  "Verify blockchain connection is established"

run_test "Blockchain Transaction" \
  "curl -s -f -X POST ${API_URL}/blockchain/transaction -H 'Content-Type: application/json' -d '{\"data\":\"test transaction\"}' | grep -q 'success'" \
  "Verify blockchain transactions work correctly"

# 5. WebSocket Tests
run_test "WebSocket Connection" \
  "npm run test:websocket" \
  "Verify WebSocket connections work correctly"

# 6. Database Tests
run_test "Database Connection" \
  "curl -s -f ${API_URL}/db/status | grep -q 'connected'" \
  "Verify database connection is established"

# 7. Integration Tests
run_test "End-to-End Threat Detection" \
  "npm run test:e2e:threat-detection" \
  "Verify end-to-end threat detection workflow"

run_test "End-to-End Alert System" \
  "npm run test:e2e:alert-system" \
  "Verify end-to-end alert system workflow"

# 8. Performance Tests
run_test "API Response Time" \
  "curl -s -w '%{time_total}' -o /dev/null ${API_URL}/health | awk '{if ($1 < 1.0) exit 0; else exit 1}'" \
  "Verify API response time is under 1 second"

run_test "AI Model Performance" \
  "npm run test:performance:ai-model" \
  "Verify AI model performance meets requirements"

# 9. Security Tests
run_test "API Security Headers" \
  "curl -s -I ${API_URL}/health | grep -q 'X-Content-Type-Options: nosniff'" \
  "Verify proper security headers are present"

run_test "SQL Injection Prevention" \
  "npm run test:security:sql-injection" \
  "Verify protection against SQL injection attacks"

# 10. Load Tests
run_test "API Load Test" \
  "npm run test:load:api" \
  "Verify API handles load correctly"

# Print test summary
echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}Test Summary${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo -e "Total Tests: ${TOTAL}"
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo -e "Success Rate: $(( (PASSED * 100) / TOTAL ))%"

# Add summary to report
echo "Test Summary" >> "$REPORT_FILE"
echo "===========" >> "$REPORT_FILE"
echo "Total Tests: $TOTAL" >> "$REPORT_FILE"
echo "Passed: $PASSED" >> "$REPORT_FILE"
echo "Failed: $FAILED" >> "$REPORT_FILE"
echo "Success Rate: $(( (PASSED * 100) / TOTAL ))%" >> "$REPORT_FILE"

echo -e "${YELLOW}Complete test report saved to: ${REPORT_FILE}${NC}" 