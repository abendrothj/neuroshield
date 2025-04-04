#!/bin/bash

# NeuraShield Performance Benchmarking Script
# This script performs comprehensive performance testing on the NeuraShield platform

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default variables
API_URL="http://localhost:3001"
ITERATIONS=100
CONCURRENCY=10
REPORT_FILE="performance-report-$(date +%Y%m%d_%H%M%S).log"
VERBOSE=false
TEST_DURATION=60 # seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --duration)
      TEST_DURATION="$2"
      shift 2
      ;;
    --report)
      REPORT_FILE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: performance-benchmark.sh [options]"
      echo "Options:"
      echo "  --api-url URL          Specify API URL (default: http://localhost:3001)"
      echo "  --iterations N         Number of requests to perform (default: 100)"
      echo "  --concurrency N        Number of concurrent requests (default: 10)"
      echo "  --duration N           Test duration in seconds (default: 60)"
      echo "  --report FILE          Specify report file name"
      echo "  --verbose              Show detailed output"
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
echo "NeuraShield Performance Benchmark Report - $(date)" > "$REPORT_FILE"
echo "=================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Test Parameters:" >> "$REPORT_FILE"
echo "- API URL: $API_URL" >> "$REPORT_FILE"
echo "- Iterations: $ITERATIONS" >> "$REPORT_FILE"
echo "- Concurrency: $CONCURRENCY" >> "$REPORT_FILE"
echo "- Test Duration: $TEST_DURATION seconds" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Check required tools
check_requirements() {
  echo -e "${YELLOW}Checking requirements...${NC}"
  
  # Check if ab (Apache Benchmark) is installed
  if ! command -v ab &> /dev/null; then
    echo -e "${RED}Error: Apache Benchmark (ab) is not installed.${NC}"
    echo "Please install it with: apt-get install apache2-utils"
    exit 1
  fi
  
  # Check if curl is installed
  if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed.${NC}"
    echo "Please install it with: apt-get install curl"
    exit 1
  fi
  
  # Check if jq is installed
  if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed.${NC}"
    echo "Please install it with: apt-get install jq"
    exit 1
  fi
  
  echo -e "${GREEN}All requirements satisfied.${NC}"
}

# Function to run a benchmark test
run_benchmark() {
  local endpoint="$1"
  local method="$2"
  local data="$3"
  local test_name="$4"
  local description="$5"
  
  echo -e "${YELLOW}Running benchmark: ${test_name}${NC}"
  echo "Benchmark: $test_name - $description" >> "$REPORT_FILE"
  
  local temp_file=$(mktemp)
  local result
  
  if [[ "$method" == "GET" ]]; then
    result=$(ab -n "$ITERATIONS" -c "$CONCURRENCY" -g "$temp_file" "$API_URL$endpoint" 2>&1)
  else
    # Create a temporary file with the POST data
    local post_data_file=$(mktemp)
    echo "$data" > "$post_data_file"
    
    result=$(ab -n "$ITERATIONS" -c "$CONCURRENCY" -p "$post_data_file" -T "application/json" -g "$temp_file" "$API_URL$endpoint" 2>&1)
    
    rm "$post_data_file"
  fi
  
  # Extract key metrics
  requests_per_second=$(echo "$result" | grep "Requests per second" | awk '{print $4}')
  time_per_request=$(echo "$result" | grep "Time per request" | head -1 | awk '{print $4}')
  transfer_rate=$(echo "$result" | grep "Transfer rate" | awk '{print $3}')
  failed_requests=$(echo "$result" | grep "Failed requests" | awk '{print $3}')
  
  # Add results to report
  echo "Results:" >> "$REPORT_FILE"
  echo "- Requests per second: $requests_per_second" >> "$REPORT_FILE"
  echo "- Time per request: $time_per_request ms" >> "$REPORT_FILE"
  echo "- Transfer rate: $transfer_rate Kbytes/sec" >> "$REPORT_FILE"
  echo "- Failed requests: $failed_requests" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  # Display results
  echo -e "${BLUE}Results:${NC}"
  echo -e "- Requests per second: $requests_per_second"
  echo -e "- Time per request: $time_per_request ms"
  echo -e "- Transfer rate: $transfer_rate Kbytes/sec"
  echo -e "- Failed requests: $failed_requests"
  
  # Save detailed results if verbose
  if [[ "$VERBOSE" == "true" ]]; then
    echo "$result" >> "${REPORT_FILE}.detailed"
    echo -e "${BLUE}Detailed results saved to ${REPORT_FILE}.detailed${NC}"
  fi
  
  # Clean up temp file
  rm "$temp_file"
  
  echo ""
}

# Function to run AI model performance test
run_ai_model_benchmark() {
  echo -e "${YELLOW}Running AI model performance benchmark...${NC}"
  echo "AI Model Performance Benchmark" >> "$REPORT_FILE"
  
  # Create a temporary file with test data
  local test_data_file=$(mktemp)
  echo '{"input":"benchmark test data for threat detection"}' > "$test_data_file"
  
  # Measure average prediction time
  local start_time=$(date +%s.%N)
  local total_predictions=30
  local successful_predictions=0
  
  for i in $(seq 1 $total_predictions); do
    if curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/ai/predict" -H "Content-Type: application/json" -d @"$test_data_file" | grep -q "200"; then
      successful_predictions=$((successful_predictions+1))
    fi
    echo -n "."
  done
  
  local end_time=$(date +%s.%N)
  local elapsed_time=$(echo "$end_time - $start_time" | bc)
  local avg_time=$(echo "$elapsed_time / $total_predictions" | bc -l)
  local success_rate=$(echo "($successful_predictions / $total_predictions) * 100" | bc -l)
  
  # Round to 2 decimal places
  avg_time=$(printf "%.2f" $avg_time)
  success_rate=$(printf "%.2f" $success_rate)
  
  echo ""
  echo -e "${BLUE}AI Model Results:${NC}"
  echo -e "- Average prediction time: $avg_time seconds"
  echo -e "- Success rate: $success_rate%"
  
  # Add results to report
  echo "Results:" >> "$REPORT_FILE"
  echo "- Average prediction time: $avg_time seconds" >> "$REPORT_FILE"
  echo "- Success rate: $success_rate%" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  # Clean up temp file
  rm "$test_data_file"
}

# Function to run WebSocket performance test
run_websocket_benchmark() {
  echo -e "${YELLOW}Running WebSocket performance benchmark...${NC}"
  echo "WebSocket Performance Benchmark" >> "$REPORT_FILE"
  
  # Run the WebSocket benchmark script (this would be a Node.js script in a real scenario)
  echo "Running WebSocket benchmark with $CONCURRENCY concurrent connections for $TEST_DURATION seconds..."
  
  # This is a placeholder - in a real scenario, you would call a specialized WebSocket benchmarking tool
  local ws_result=$(node scripts/websocket-benchmark.js --url "ws://${API_URL#http://}/ws" --connections "$CONCURRENCY" --duration "$TEST_DURATION" 2>&1 || echo "WebSocket benchmark failed")
  
  # Add results to report
  echo "Results:" >> "$REPORT_FILE"
  echo "$ws_result" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  # Display results
  echo -e "${BLUE}WebSocket Results:${NC}"
  echo "$ws_result"
}

# Function to run database performance test
run_database_benchmark() {
  echo -e "${YELLOW}Running database performance benchmark...${NC}"
  echo "Database Performance Benchmark" >> "$REPORT_FILE"
  
  # Create a temporary file with test data
  local test_data_file=$(mktemp)
  echo '{"query":"SELECT * FROM test_collection LIMIT 100"}' > "$test_data_file"
  
  # Measure average query time
  local start_time=$(date +%s.%N)
  local total_queries=20
  local successful_queries=0
  
  for i in $(seq 1 $total_queries); do
    if curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/db/query" -H "Content-Type: application/json" -d @"$test_data_file" | grep -q "200"; then
      successful_queries=$((successful_queries+1))
    fi
    echo -n "."
  done
  
  local end_time=$(date +%s.%N)
  local elapsed_time=$(echo "$end_time - $start_time" | bc)
  local avg_time=$(echo "$elapsed_time / $total_queries" | bc -l)
  local success_rate=$(echo "($successful_queries / $total_queries) * 100" | bc -l)
  
  # Round to 2 decimal places
  avg_time=$(printf "%.2f" $avg_time)
  success_rate=$(printf "%.2f" $success_rate)
  
  echo ""
  echo -e "${BLUE}Database Results:${NC}"
  echo -e "- Average query time: $avg_time seconds"
  echo -e "- Success rate: $success_rate%"
  
  # Add results to report
  echo "Results:" >> "$REPORT_FILE"
  echo "- Average query time: $avg_time seconds" >> "$REPORT_FILE"
  echo "- Success rate: $success_rate%" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  # Clean up temp file
  rm "$test_data_file"
}

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}NeuraShield Performance Benchmark${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# Check requirements
check_requirements

# API endpoint benchmarks
echo -e "${YELLOW}Running API endpoint benchmarks...${NC}"

# 1. Health endpoint
run_benchmark "/health" "GET" "" "Health Endpoint" "Basic health check endpoint"

# 2. Authentication endpoint
run_benchmark "/auth/login" "POST" '{"username":"test_user","password":"test_password"}' "Authentication" "User login endpoint"

# 3. AI prediction endpoint
run_benchmark "/ai/predict" "POST" '{"input":"test data for threat detection"}' "AI Prediction" "Threat detection prediction endpoint"

# 4. Blockchain status endpoint
run_benchmark "/blockchain/status" "GET" "" "Blockchain Status" "Blockchain connection status endpoint"

# 5. Event retrieval endpoint
run_benchmark "/events" "GET" "" "Events Retrieval" "Security events retrieval endpoint"

# AI model performance benchmark
run_ai_model_benchmark

# WebSocket performance benchmark
run_websocket_benchmark

# Database performance benchmark
run_database_benchmark

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}Benchmark Summary${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo -e "Performance benchmark completed."
echo -e "Full report saved to: ${REPORT_FILE}"

# Add timestamp to report
echo "" >> "$REPORT_FILE"
echo "Benchmark completed at $(date)" >> "$REPORT_FILE"

# Return success
exit 0 