#!/bin/bash

# NeuraShield Comprehensive System Test
# This script performs a full system test covering API, AI, blockchain, WebSocket, etc.

# Source the shared test utilities
source "$(dirname "$0")/test-utils.sh"

# Command line arguments
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
    --help)
      echo "Usage: system-test.sh [options]"
      echo "Options:"
      echo "  --api-url URL      Specify API URL (default: $API_URL)"
      echo "  --verbose          Show detailed output for all tests"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/system-test-report-$TIMESTAMP.log"

# Initialize report file
echo "NeuraShield System Test Report - $(date)" > "$REPORT_FILE"
echo "=======================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Test Environment:" >> "$REPORT_FILE"
echo "- API URL: $API_URL" >> "$REPORT_FILE"
echo "- Timestamp: $TIMESTAMP" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}NeuraShield Comprehensive System Test${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# Check if services are online
check_services "$REPORT_FILE"

# Testing section
echo "" >> "$REPORT_FILE"
echo "System Tests" >> "$REPORT_FILE"
echo "============" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 1. API Health Check
run_test "API Health Check" \
  "curl -s -f ${API_URL}/health | grep -q 'status.*ok'" \
  "Verify API is accessible and healthy" \
  "high" \
  "$REPORT_FILE"

# 2. Authentication Tests
run_test "User Authentication" \
  "curl -s -f -X POST ${API_URL}/auth/login -H 'Content-Type: application/json' -d '{\"username\":\"test_user\",\"password\":\"test_password\"}' | grep -q 'token'" \
  "Verify user authentication works correctly" \
  "high" \
  "$REPORT_FILE"

# 3. AI Model Tests
run_test "AI Model Status" \
  "curl -s -f ${API_URL}/ai/status | grep -q 'running'" \
  "Verify AI model service is running" \
  "high" \
  "$REPORT_FILE"

run_test "AI Model Prediction" \
  "curl -s -f -X POST ${API_URL}/ai/predict -H 'Content-Type: application/json' -d '{\"input\":\"test data\"}' | grep -q 'prediction'" \
  "Verify AI model can make predictions" \
  "high" \
  "$REPORT_FILE"

# 4. Blockchain Tests
run_test "Blockchain Connection" \
  "curl -s -f ${API_URL}/blockchain/status | grep -q 'connected'" \
  "Verify blockchain connection is established" \
  "medium" \
  "$REPORT_FILE"

run_test "Blockchain Transaction" \
  "curl -s -f -X POST ${API_URL}/blockchain/transaction -H 'Content-Type: application/json' -d '{\"data\":\"test transaction\"}' | grep -q 'success'" \
  "Verify blockchain transactions work correctly" \
  "medium" \
  "$REPORT_FILE"

# 5. WebSocket Tests
run_test "WebSocket Connection" \
  "npm run test:websocket" \
  "Verify WebSocket connections work correctly" \
  "medium" \
  "$REPORT_FILE"

# 6. Database Tests
run_test "Database Connection" \
  "curl -s -f ${API_URL}/db/status | grep -q 'connected'" \
  "Verify database connection is established" \
  "high" \
  "$REPORT_FILE"

# 7. Performance Tests
run_test "API Response Time" \
  "curl -s -w '%{time_total}' -o /dev/null ${API_URL}/health | awk '{if ($1 < 1.0) exit 0; else exit 1}'" \
  "Verify API response time is under 1 second" \
  "low" \
  "$REPORT_FILE"

run_test "AI Model Performance" \
  "npm run test:performance:ai-model" \
  "Verify AI model performance meets requirements" \
  "medium" \
  "$REPORT_FILE"

# 8. Security Tests
run_test "API Security Headers" \
  "curl -s -I ${API_URL}/health | grep -q 'X-Content-Type-Options: nosniff'" \
  "Verify proper security headers are present" \
  "medium" \
  "$REPORT_FILE"

run_test "SQL Injection Prevention" \
  "npm run test:security:sql-injection" \
  "Verify protection against SQL injection attacks" \
  "high" \
  "$REPORT_FILE"

# 9. Load Tests
run_test "API Load Test" \
  "npm run test:load:api" \
  "Verify API handles load correctly" \
  "low" \
  "$REPORT_FILE"

# Print test summary
print_test_summary "$REPORT_FILE" 