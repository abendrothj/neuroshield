#!/bin/bash

# Shared utility functions for NeuraShield test scripts

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default variables
API_URL="http://localhost:3001"
FRONTEND_URL="http://localhost:3000"
AI_URL="http://localhost:5000"
REPORT_DIR="reports"
VERBOSE=false

# Initialize counters
TOTAL=0
PASSED=0
FAILED=0
WARNING=0
SKIPPED=0

# Ensure report directory exists
mkdir -p "$REPORT_DIR"

# Function to run a test and record results
run_test() {
  local test_name="$1"
  local test_cmd="$2"
  local test_desc="$3"
  local severity="${4:-normal}"  # optional: high, medium, low, normal
  local report_file="$5"
  
  echo -e "${YELLOW}Running test: ${test_name} (${severity} severity)${NC}"
  
  if [[ -n "$report_file" ]]; then
    echo "Test: $test_name - $test_desc" >> "$report_file"
    echo "Severity: $severity" >> "$report_file"
  fi
  
  TOTAL=$((TOTAL+1))
  
  # Run the test command and capture output
  local output
  local exit_code
  
  output=$(eval "$test_cmd" 2>&1) || exit_code=$?
  
  if [[ -z "$exit_code" ]]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    PASSED=$((PASSED+1))
    
    if [[ -n "$report_file" ]]; then
      echo "Status: PASSED" >> "$report_file"
    fi
  else
    if [[ "$severity" == "high" ]]; then
      echo -e "${RED}✗ FAILED${NC}"
      FAILED=$((FAILED+1))
      
      if [[ -n "$report_file" ]]; then
        echo "Status: FAILED" >> "$report_file"
      fi
    else
      echo -e "${YELLOW}⚠ WARNING${NC}"
      WARNING=$((WARNING+1))
      
      if [[ -n "$report_file" ]]; then
        echo "Status: WARNING" >> "$report_file"
      fi
    fi
  fi
  
  if [[ "$VERBOSE" == "true" || ! -z "$exit_code" ]]; then
    echo -e "${BLUE}Output:${NC}"
    echo "$output"
    
    if [[ -n "$report_file" ]]; then
      echo "Output:" >> "$report_file"
      echo "$output" >> "$report_file"
    fi
  fi
  
  if [[ -n "$report_file" ]]; then
    echo "" >> "$report_file"
  fi
}

# Function to run a manual user test
run_manual_test() {
  local test_name="$1"
  local test_desc="$2"
  local test_steps="$3"
  local expected_result="$4"
  local report_file="$5"
  
  echo -e "${YELLOW}Manual Test: ${test_name}${NC}"
  echo -e "${BLUE}Description:${NC} $test_desc"
  echo -e "${BLUE}Steps:${NC}"
  echo "$test_steps"
  echo -e "${BLUE}Expected Result:${NC} $expected_result"
  
  if [[ -n "$report_file" ]]; then
    echo "Test: $test_name - $test_desc" >> "$report_file"
    echo "Steps: $test_steps" >> "$report_file"
    echo "Expected Result: $expected_result" >> "$report_file"
  fi
  
  TOTAL=$((TOTAL+1))
  
  # Prompt for test result
  while true; do
    echo -e "${YELLOW}Did the test pass? (y/n/s - skip)${NC}"
    read -r result
    
    if [[ "$result" == "y" || "$result" == "Y" ]]; then
      echo -e "${GREEN}✓ PASSED${NC}"
      PASSED=$((PASSED+1))
      
      if [[ -n "$report_file" ]]; then
        echo "Status: PASSED" >> "$report_file"
      fi
      break
    elif [[ "$result" == "n" || "$result" == "N" ]]; then
      echo -e "${RED}✗ FAILED${NC}"
      echo -e "${YELLOW}Please enter a description of the issue:${NC}"
      read -r issue_desc
      FAILED=$((FAILED+1))
      
      if [[ -n "$report_file" ]]; then
        echo "Status: FAILED" >> "$report_file"
        echo "Issue: $issue_desc" >> "$report_file"
      fi
      break
    elif [[ "$result" == "s" || "$result" == "S" ]]; then
      echo -e "${BLUE}⚠ SKIPPED${NC}"
      SKIPPED=$((SKIPPED+1))
      
      if [[ -n "$report_file" ]]; then
        echo "Status: SKIPPED" >> "$report_file"
      fi
      break
    else
      echo -e "${RED}Invalid input. Please enter 'y', 'n', or 's'.${NC}"
    fi
  done
  
  if [[ -n "$report_file" ]]; then
    echo "" >> "$report_file"
  fi
}

# Function to print test summary
print_test_summary() {
  local report_file="$1"
  
  echo -e "${YELLOW}=================================================${NC}"
  echo -e "${YELLOW}Test Summary${NC}"
  echo -e "${YELLOW}=================================================${NC}"
  echo -e "Total Tests: ${TOTAL}"
  echo -e "Passed: ${GREEN}${PASSED}${NC}"
  echo -e "Failed: ${RED}${FAILED}${NC}"
  
  if [[ $WARNING -gt 0 ]]; then
    echo -e "Warnings: ${YELLOW}${WARNING}${NC}"
  fi
  
  if [[ $SKIPPED -gt 0 ]]; then
    echo -e "Skipped: ${BLUE}${SKIPPED}${NC}"
    echo -e "Success Rate: $(( (PASSED * 100) / (TOTAL - SKIPPED) ))%"
  else
    echo -e "Success Rate: $(( (PASSED * 100) / TOTAL ))%"
  fi

  if [[ -n "$report_file" ]]; then
    # Add summary to report
    echo "" >> "$report_file"
    echo "Test Summary" >> "$report_file"
    echo "===========" >> "$report_file"
    echo "Total Tests: $TOTAL" >> "$report_file"
    echo "Passed: $PASSED" >> "$report_file"
    echo "Failed: $FAILED" >> "$report_file"
    
    if [[ $WARNING -gt 0 ]]; then
      echo "Warnings: $WARNING" >> "$report_file"
    fi
    
    if [[ $SKIPPED -gt 0 ]]; then
      echo "Skipped: $SKIPPED" >> "$report_file"
      echo "Success Rate: $(( (PASSED * 100) / (TOTAL - SKIPPED) ))%" >> "$report_file"
    else
      echo "Success Rate: $(( (PASSED * 100) / TOTAL ))%" >> "$report_file"
    fi
    
    echo -e "${YELLOW}Complete test report saved to: ${report_file}${NC}"
  fi
}

# Function to check if services are online
check_services() {
  local report_file="$1"
  
  echo -e "${YELLOW}Checking if services are online...${NC}"
  
  if [[ -n "$report_file" ]]; then
    echo "Service Status Check" >> "$report_file"
    echo "===================" >> "$report_file"
  fi
  
  # Check API
  if curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" | grep -q "200"; then
    echo -e "${GREEN}✓ API service is online${NC}"
    
    if [[ -n "$report_file" ]]; then
      echo "API service: ONLINE" >> "$report_file"
    fi
  else
    echo -e "${RED}✗ API service is offline or not responding${NC}"
    
    if [[ -n "$report_file" ]]; then
      echo "API service: OFFLINE" >> "$report_file"
    fi
  fi
  
  # Check Frontend
  if curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL" | grep -q "200"; then
    echo -e "${GREEN}✓ Frontend service is online${NC}"
    
    if [[ -n "$report_file" ]]; then
      echo "Frontend service: ONLINE" >> "$report_file"
    fi
  else
    echo -e "${RED}✗ Frontend service is offline or not responding${NC}"
    
    if [[ -n "$report_file" ]]; then
      echo "Frontend service: OFFLINE" >> "$report_file"
    fi
  fi
  
  # Check AI Service
  if curl -s -o /dev/null -w "%{http_code}" "$AI_URL/health" | grep -q "200"; then
    echo -e "${GREEN}✓ AI service is online${NC}"
    
    if [[ -n "$report_file" ]]; then
      echo "AI service: ONLINE" >> "$report_file"
    fi
  else
    echo -e "${RED}✗ AI service is offline or not responding${NC}"
    
    if [[ -n "$report_file" ]]; then
      echo "AI service: OFFLINE" >> "$report_file"
    fi
  fi
  
  if [[ -n "$report_file" ]]; then
    echo "" >> "$report_file"
  fi
  
  echo ""
} 