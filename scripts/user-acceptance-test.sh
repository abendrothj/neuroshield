#!/bin/bash

# NeuraShield User Acceptance Testing Script
# This script guides through manual user acceptance testing with automated checks

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default variables
API_URL="http://localhost:3001"
FRONTEND_URL="http://localhost:3000"
TEST_USERNAME="test_user"
TEST_PASSWORD="test_password"
REPORT_FILE="uat-report-$(date +%Y%m%d_%H%M%S).log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --frontend-url)
      FRONTEND_URL="$2"
      shift 2
      ;;
    --username)
      TEST_USERNAME="$2"
      shift 2
      ;;
    --password)
      TEST_PASSWORD="$2"
      shift 2
      ;;
    --report)
      REPORT_FILE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: user-acceptance-test.sh [options]"
      echo "Options:"
      echo "  --api-url URL          Specify API URL (default: http://localhost:3001)"
      echo "  --frontend-url URL     Specify frontend URL (default: http://localhost:3000)"
      echo "  --username USER        Test username (default: test_user)"
      echo "  --password PASS        Test password (default: test_password)"
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
echo "NeuraShield User Acceptance Test Report - $(date)" > "$REPORT_FILE"
echo "===============================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Test Environment:" >> "$REPORT_FILE"
echo "- API URL: $API_URL" >> "$REPORT_FILE"
echo "- Frontend URL: $FRONTEND_URL" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Test results tracking
PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

# Function to check if service is online
check_service() {
  echo -e "${YELLOW}Checking if services are online...${NC}"
  
  # Check API
  if curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" | grep -q "200"; then
    echo -e "${GREEN}✓ API service is online${NC}"
    echo "API service: ONLINE" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ API service is offline or not responding${NC}"
    echo "API service: OFFLINE" >> "$REPORT_FILE"
    echo "Cannot proceed with testing. Please ensure API service is running."
    exit 1
  fi
  
  # Check Frontend (just check if the page returns 200)
  if curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL" | grep -q "200"; then
    echo -e "${GREEN}✓ Frontend service is online${NC}"
    echo "Frontend service: ONLINE" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ Frontend service is offline or not responding${NC}"
    echo "Frontend service: OFFLINE" >> "$REPORT_FILE"
    echo "Cannot proceed with testing. Please ensure Frontend service is running."
    exit 1
  fi
  
  echo ""
}

# Function to run an automated test
run_automated_test() {
  local test_name="$1"
  local test_cmd="$2"
  local test_desc="$3"
  
  echo -e "${YELLOW}Running automated test: ${test_name}${NC}"
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
    echo "Output: $output" >> "$REPORT_FILE"
  fi
  
  echo ""
}

# Function to run a manual user test
run_manual_test() {
  local test_name="$1"
  local test_desc="$2"
  local test_steps="$3"
  local expected_result="$4"
  
  echo -e "${YELLOW}Manual Test: ${test_name}${NC}"
  echo -e "${BLUE}Description:${NC} $test_desc"
  echo -e "${BLUE}Steps:${NC}"
  echo "$test_steps"
  echo -e "${BLUE}Expected Result:${NC} $expected_result"
  
  echo "Test: $test_name - $test_desc" >> "$REPORT_FILE"
  echo "Steps: $test_steps" >> "$REPORT_FILE"
  echo "Expected Result: $expected_result" >> "$REPORT_FILE"
  
  TOTAL=$((TOTAL+1))
  
  # Prompt for test result
  while true; do
    echo -e "${YELLOW}Did the test pass? (y/n/s - skip)${NC}"
    read -r result
    
    if [[ "$result" == "y" || "$result" == "Y" ]]; then
      echo -e "${GREEN}✓ PASSED${NC}"
      PASSED=$((PASSED+1))
      echo "Status: PASSED" >> "$REPORT_FILE"
      break
    elif [[ "$result" == "n" || "$result" == "N" ]]; then
      echo -e "${RED}✗ FAILED${NC}"
      echo -e "${YELLOW}Please enter a description of the issue:${NC}"
      read -r issue_desc
      FAILED=$((FAILED+1))
      echo "Status: FAILED" >> "$REPORT_FILE"
      echo "Issue: $issue_desc" >> "$REPORT_FILE"
      break
    elif [[ "$result" == "s" || "$result" == "S" ]]; then
      echo -e "${BLUE}⚠ SKIPPED${NC}"
      SKIPPED=$((SKIPPED+1))
      echo "Status: SKIPPED" >> "$REPORT_FILE"
      break
    else
      echo -e "${RED}Invalid input. Please enter 'y', 'n', or 's'.${NC}"
    fi
  done
  
  echo ""
}

# Main UAT Function
run_user_acceptance_testing() {
  echo -e "${YELLOW}=================================================${NC}"
  echo -e "${YELLOW}NeuraShield User Acceptance Testing${NC}"
  echo -e "${YELLOW}=================================================${NC}"
  echo ""
  
  # Check if services are online
  check_service
  
  # Automated Tests
  echo -e "${YELLOW}Running automated tests...${NC}"
  echo "" >> "$REPORT_FILE"
  echo "Automated Tests" >> "$REPORT_FILE"
  echo "===============" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  # 1. Login Test
  run_automated_test "Login API" \
    "curl -s -X POST '$API_URL/auth/login' -H 'Content-Type: application/json' -d '{\"username\":\"$TEST_USERNAME\",\"password\":\"$TEST_PASSWORD\"}' | grep -q 'token'" \
    "Verify that login API returns a token"
  
  # 2. API Data Retrieval Test
  run_automated_test "Data Retrieval" \
    "curl -s '$API_URL/events' | grep -q 'data'" \
    "Verify that events API returns data"
  
  # 3. AI Model Prediction Test
  run_automated_test "AI Prediction" \
    "curl -s -X POST '$API_URL/ai/predict' -H 'Content-Type: application/json' -d '{\"input\":\"test data\"}' | grep -q 'prediction'" \
    "Verify that AI prediction API returns results"
  
  # Manual Tests
  echo -e "${YELLOW}Running manual tests...${NC}"
  echo "" >> "$REPORT_FILE"
  echo "Manual Tests" >> "$REPORT_FILE"
  echo "============" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  # 1. User Login Test
  run_manual_test "User Login" \
    "Verify that a user can log in to the application" \
    "1. Open $FRONTEND_URL in your browser\n2. Enter username: $TEST_USERNAME\n3. Enter password: $TEST_PASSWORD\n4. Click the Login button" \
    "User should be logged in and redirected to the dashboard page"
  
  # 2. Dashboard Display Test
  run_manual_test "Dashboard Display" \
    "Verify that the dashboard displays threat metrics and visualizations" \
    "1. Log in to the application\n2. Navigate to the Dashboard page" \
    "Dashboard should display threat metrics, charts, and real-time updates"
  
  # 3. AI Monitoring Test
  run_manual_test "AI Monitoring" \
    "Verify that the AI monitoring page shows model performance" \
    "1. Log in to the application\n2. Navigate to the AI Monitoring page" \
    "Page should display model performance metrics, accuracy charts, and training history"
  
  # 4. Event Timeline Test
  run_manual_test "Event Timeline" \
    "Verify that the event timeline shows security events" \
    "1. Log in to the application\n2. Navigate to the Events page" \
    "Page should display a timeline of security events with details and filtering options"
  
  # 5. Model Training Test
  run_manual_test "Model Training" \
    "Verify that users can initiate model training" \
    "1. Log in to the application\n2. Navigate to the Model Training page\n3. Configure training parameters\n4. Start training" \
    "Training should start and display progress indicators"
  
  # 6. Settings Configuration Test
  run_manual_test "Settings Configuration" \
    "Verify that users can update system settings" \
    "1. Log in to the application\n2. Navigate to the Settings page\n3. Modify notification settings\n4. Save changes" \
    "Settings should be saved and a confirmation message displayed"
  
  # 7. Real-time Update Test
  run_manual_test "Real-time Updates" \
    "Verify that the system displays real-time updates" \
    "1. Log in to the application\n2. Navigate to the Dashboard\n3. Wait for new events to appear (or trigger a test event)" \
    "New events should appear in real-time without page refresh"
  
  # 8. Mobile Responsiveness Test
  run_manual_test "Mobile Responsiveness" \
    "Verify that the application is responsive on mobile devices" \
    "1. Open the application on a mobile device or using browser responsive mode\n2. Navigate through different pages" \
    "UI should adapt to different screen sizes and remain usable"
  
  # 9. Blockchain Verification Test
  run_manual_test "Blockchain Verification" \
    "Verify that security events are properly recorded on the blockchain" \
    "1. Log in to the application\n2. Navigate to the Events page\n3. Select an event and view its blockchain verification details" \
    "Event should show blockchain transaction details and verification status"
  
  # 10. User Management Test
  run_manual_test "User Management" \
    "Verify that administrators can manage users" \
    "1. Log in as an administrator\n2. Navigate to the User Management page\n3. Create a new test user\n4. Edit user permissions\n5. Delete the test user" \
    "All user management operations should work correctly with appropriate feedback"
}

# Run the user acceptance testing
run_user_acceptance_testing

# Print summary
echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}User Acceptance Test Summary${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo -e "Total Tests: ${TOTAL}"
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo -e "Skipped: ${BLUE}${SKIPPED}${NC}"
echo -e "Success Rate: $(( (PASSED * 100) / (TOTAL - SKIPPED) ))%"

# Add summary to report
echo "" >> "$REPORT_FILE"
echo "User Acceptance Test Summary" >> "$REPORT_FILE"
echo "===========================" >> "$REPORT_FILE"
echo "Total Tests: $TOTAL" >> "$REPORT_FILE"
echo "Passed: $PASSED" >> "$REPORT_FILE"
echo "Failed: $FAILED" >> "$REPORT_FILE"
echo "Skipped: $SKIPPED" >> "$REPORT_FILE"
echo "Success Rate: $(( (PASSED * 100) / (TOTAL - SKIPPED) ))%" >> "$REPORT_FILE"

echo -e "${YELLOW}Complete UAT report saved to: ${REPORT_FILE}${NC}" 