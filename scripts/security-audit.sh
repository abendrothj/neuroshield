#!/bin/bash

# NeuraShield Security Audit Script
# This script performs a comprehensive security audit on the NeuraShield platform

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
TARGET_IP="127.0.0.1"
REPORT_FILE="security-audit-$(date +%Y%m%d_%H%M%S).log"
VERBOSE=false
SKIP_INTRUSIVE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --target)
      TARGET_IP="$2"
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
    --include-intrusive)
      SKIP_INTRUSIVE=false
      shift
      ;;
    --help)
      echo "Usage: security-audit.sh [options]"
      echo "Options:"
      echo "  --api-url URL          Specify API URL (default: http://localhost:3001)"
      echo "  --target IP            Target IP address (default: 127.0.0.1)"
      echo "  --report FILE          Specify report file name"
      echo "  --verbose              Show detailed output"
      echo "  --include-intrusive    Include potentially intrusive tests"
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
echo "NeuraShield Security Audit Report - $(date)" > "$REPORT_FILE"
echo "============================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Target Information:" >> "$REPORT_FILE"
echo "- API URL: $API_URL" >> "$REPORT_FILE"
echo "- Target IP: $TARGET_IP" >> "$REPORT_FILE"
echo "- Intrusive Tests: $(if [[ "$SKIP_INTRUSIVE" == "true" ]]; then echo "Disabled"; else echo "Enabled"; fi)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Test results tracking
PASSED=0
FAILED=0
WARNING=0
TOTAL=0

# Check required tools
check_requirements() {
  echo -e "${YELLOW}Checking requirements...${NC}"
  
  # Check if curl is installed
  if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed.${NC}"
    echo "Please install it with: apt-get install curl"
    exit 1
  fi
  
  # Check if nmap is installed
  if ! command -v nmap &> /dev/null; then
    echo -e "${RED}Error: nmap is not installed.${NC}"
    echo "Please install it with: apt-get install nmap"
    exit 1
  fi
  
  # Check if nikto is installed (optional)
  if ! command -v nikto &> /dev/null; then
    echo -e "${YELLOW}Warning: nikto is not installed. Web vulnerability scanning will be limited.${NC}"
    echo "You can install it with: apt-get install nikto"
  fi
  
  # Check if sslyze is installed (optional)
  if ! command -v sslyze &> /dev/null; then
    echo -e "${YELLOW}Warning: sslyze is not installed. SSL/TLS testing will be limited.${NC}"
    echo "You can install it with: pip install sslyze"
  fi
  
  echo -e "${GREEN}Requirements check completed.${NC}"
}

# Function to run a security test
run_test() {
  local test_name="$1"
  local test_cmd="$2"
  local test_desc="$3"
  local severity="$4"  # high, medium, low
  
  echo -e "${YELLOW}Running test: ${test_name} (${severity} severity)${NC}"
  echo "Test: $test_name - $test_desc" >> "$REPORT_FILE"
  echo "Severity: $severity" >> "$REPORT_FILE"
  
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
    if [[ "$severity" == "high" ]]; then
      echo -e "${RED}✗ FAILED${NC}"
      FAILED=$((FAILED+1))
      echo "Status: FAILED" >> "$REPORT_FILE"
    else
      echo -e "${YELLOW}⚠ WARNING${NC}"
      WARNING=$((WARNING+1))
      echo "Status: WARNING" >> "$REPORT_FILE"
    fi
  fi
  
  if [[ "$VERBOSE" == "true" || ! -z "$exit_code" ]]; then
    echo -e "${BLUE}Output:${NC}"
    echo "$output"
    echo "Output:" >> "$REPORT_FILE"
    echo "$output" >> "$REPORT_FILE"
  fi
  
  echo "" >> "$REPORT_FILE"
}

# Function to check HTTP security headers
check_security_headers() {
  echo -e "${YELLOW}Checking HTTP security headers...${NC}"
  echo "HTTP Security Headers Check" >> "$REPORT_FILE"
  
  local headers=$(curl -s -I "$API_URL/health")
  
  # Check for X-Content-Type-Options
  if echo "$headers" | grep -q "X-Content-Type-Options: nosniff"; then
    echo -e "${GREEN}✓ X-Content-Type-Options: nosniff is set${NC}"
    echo "X-Content-Type-Options: PASSED" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ X-Content-Type-Options header is missing${NC}"
    echo "X-Content-Type-Options: FAILED" >> "$REPORT_FILE"
  fi
  
  # Check for X-Frame-Options
  if echo "$headers" | grep -q "X-Frame-Options: DENY\|X-Frame-Options: SAMEORIGIN"; then
    echo -e "${GREEN}✓ X-Frame-Options is set${NC}"
    echo "X-Frame-Options: PASSED" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ X-Frame-Options header is missing${NC}"
    echo "X-Frame-Options: FAILED" >> "$REPORT_FILE"
  fi
  
  # Check for Content-Security-Policy
  if echo "$headers" | grep -q "Content-Security-Policy:"; then
    echo -e "${GREEN}✓ Content-Security-Policy is set${NC}"
    echo "Content-Security-Policy: PASSED" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ Content-Security-Policy header is missing${NC}"
    echo "Content-Security-Policy: FAILED" >> "$REPORT_FILE"
  fi
  
  # Check for Strict-Transport-Security
  if echo "$headers" | grep -q "Strict-Transport-Security:"; then
    echo -e "${GREEN}✓ Strict-Transport-Security is set${NC}"
    echo "Strict-Transport-Security: PASSED" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ Strict-Transport-Security header is missing${NC}"
    echo "Strict-Transport-Security: FAILED" >> "$REPORT_FILE"
  fi
  
  # Check for X-XSS-Protection
  if echo "$headers" | grep -q "X-XSS-Protection: 1; mode=block"; then
    echo -e "${GREEN}✓ X-XSS-Protection is set${NC}"
    echo "X-XSS-Protection: PASSED" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ X-XSS-Protection header is missing or incorrect${NC}"
    echo "X-XSS-Protection: FAILED" >> "$REPORT_FILE"
  fi
  
  echo "" >> "$REPORT_FILE"
}

# Function to perform port scanning
port_scan() {
  echo -e "${YELLOW}Performing port scan on $TARGET_IP...${NC}"
  echo "Port Scan Results" >> "$REPORT_FILE"
  
  # Perform a basic port scan
  local scan_result=$(nmap -sT -p 1-10000 --open "$TARGET_IP" -T4)
  
  echo "$scan_result" >> "$REPORT_FILE"
  
  # Display open ports
  echo -e "${BLUE}Open ports:${NC}"
  echo "$scan_result" | grep "^[0-9]" | sed 's/\/.*//g'
  
  echo "" >> "$REPORT_FILE"
}

# Function to check for SQL injection vulnerabilities
check_sql_injection() {
  echo -e "${YELLOW}Testing for SQL injection vulnerabilities...${NC}"
  echo "SQL Injection Test" >> "$REPORT_FILE"
  
  # Array of SQL injection payloads
  local payloads=(
    "' OR '1'='1"
    "1 OR 1=1"
    "1' OR '1' = '1' --"
    "1' OR '1' = '1' { -- }"
    "' UNION SELECT 1,2,3 --"
  )
  
  local vulnerable=false
  
  # Test login endpoint
  echo -e "${BLUE}Testing login endpoint for SQL injection...${NC}"
  
  for payload in "${payloads[@]}"; do
    local response=$(curl -s -X POST "$API_URL/auth/login" -H "Content-Type: application/json" -d "{\"username\":\"$payload\",\"password\":\"test\"}")
    
    # Check if the response indicates successful authentication or error
    if echo "$response" | grep -q "token\|success\|admin\|loggedin"; then
      echo -e "${RED}✗ Potential SQL injection vulnerability found with payload: $payload${NC}"
      echo "Potential vulnerability found with payload: $payload" >> "$REPORT_FILE"
      vulnerable=true
    fi
  done
  
  if [[ "$vulnerable" == "false" ]]; then
    echo -e "${GREEN}✓ No obvious SQL injection vulnerabilities detected${NC}"
    echo "No obvious SQL injection vulnerabilities detected" >> "$REPORT_FILE"
  fi
  
  echo "" >> "$REPORT_FILE"
}

# Function to check for XSS vulnerabilities
check_xss() {
  echo -e "${YELLOW}Testing for XSS vulnerabilities...${NC}"
  echo "XSS Vulnerability Test" >> "$REPORT_FILE"
  
  # Array of XSS payloads
  local payloads=(
    "<script>alert(1)</script>"
    "<img src='x' onerror='alert(1)'>"
    "\"><script>alert(1)</script>"
    "javascript:alert(1)"
  )
  
  local vulnerable=false
  
  # Test various endpoints for XSS reflection
  echo -e "${BLUE}Testing for reflected XSS...${NC}"
  
  # Testing search endpoint (example)
  for payload in "${payloads[@]}"; do
    local response=$(curl -s "$API_URL/search?q=$payload")
    
    # Check if the payload is reflected in the response
    if echo "$response" | grep -q "$payload"; then
      echo -e "${RED}✗ Potential XSS vulnerability found with payload: $payload${NC}"
      echo "Potential XSS vulnerability found with payload: $payload" >> "$REPORT_FILE"
      vulnerable=true
    fi
  done
  
  if [[ "$vulnerable" == "false" ]]; then
    echo -e "${GREEN}✓ No obvious XSS vulnerabilities detected${NC}"
    echo "No obvious XSS vulnerabilities detected" >> "$REPORT_FILE"
  fi
  
  echo "" >> "$REPORT_FILE"
}

# Function to check SSL/TLS configuration
check_ssl_tls() {
  # Extract hostname from API_URL
  local hostname=$(echo "$API_URL" | sed -e 's|^[^/]*//||' -e 's|/.*$||' -e 's|:.*$||')
  
  echo -e "${YELLOW}Checking SSL/TLS configuration for $hostname...${NC}"
  echo "SSL/TLS Configuration Check" >> "$REPORT_FILE"
  
  # Check if hostname is not localhost or 127.0.0.1
  if [[ "$hostname" == "localhost" || "$hostname" == "127.0.0.1" ]]; then
    echo -e "${YELLOW}Skipping SSL/TLS check for localhost${NC}"
    echo "Skipped SSL/TLS check for localhost" >> "$REPORT_FILE"
    return
  fi
  
  # Check if sslyze is available
  if command -v sslyze &> /dev/null; then
    local result=$(sslyze --regular "$hostname" 2>&1)
    echo "$result" >> "$REPORT_FILE"
    
    # Check for weak protocols
    if echo "$result" | grep -q "SSLv2\|SSLv3\|TLSv1.0\|TLSv1.1"; then
      echo -e "${RED}✗ Weak SSL/TLS protocols detected${NC}"
    else
      echo -e "${GREEN}✓ No weak SSL/TLS protocols detected${NC}"
    fi
    
    # Check for weak cipher suites
    if echo "$result" | grep -q "DES\|RC4\|NULL\|EXPORT\|anon"; then
      echo -e "${RED}✗ Weak cipher suites detected${NC}"
    else
      echo -e "${GREEN}✓ No weak cipher suites detected${NC}"
    fi
  else
    # Fallback to OpenSSL
    echo -e "${YELLOW}Using OpenSSL for basic SSL/TLS check${NC}"
    
    local openssl_result=$(openssl s_client -connect "$hostname:443" -tls1_2 </dev/null 2>&1)
    echo "$openssl_result" >> "$REPORT_FILE"
    
    if echo "$openssl_result" | grep -q "Cipher is"; then
      echo -e "${GREEN}✓ TLS connection successful${NC}"
    else
      echo -e "${RED}✗ TLS connection failed${NC}"
    fi
  fi
  
  echo "" >> "$REPORT_FILE"
}

# Function to check API authentication
check_api_auth() {
  echo -e "${YELLOW}Testing API authentication and authorization...${NC}"
  echo "API Authentication and Authorization Test" >> "$REPORT_FILE"
  
  # Test access to a protected endpoint without authentication
  local response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/protected-resource")
  
  if [[ "$response" == "401" || "$response" == "403" ]]; then
    echo -e "${GREEN}✓ Protected endpoint correctly requires authentication${NC}"
    echo "Protected endpoint correctly requires authentication: $response" >> "$REPORT_FILE"
  else
    echo -e "${RED}✗ Protected endpoint does not require authentication (HTTP $response)${NC}"
    echo "FAILED: Protected endpoint does not require authentication: $response" >> "$REPORT_FILE"
  fi
  
  echo "" >> "$REPORT_FILE"
}

# Main Security Audit Function
run_security_audit() {
  echo -e "${YELLOW}=================================================${NC}"
  echo -e "${YELLOW}NeuraShield Security Audit${NC}"
  echo -e "${YELLOW}=================================================${NC}"
  echo ""
  
  # Check requirements
  check_requirements
  
  # 1. Port Scanning
  port_scan
  
  # 2. HTTP Security Headers
  check_security_headers
  
  # 3. API Authentication Test
  check_api_auth
  
  # 4. SQL Injection Test
  check_sql_injection
  
  # 5. XSS Vulnerability Test
  check_xss
  
  # 6. SSL/TLS Configuration Check
  check_ssl_tls
  
  # 7. Run specific security tests
  run_test "API Rate Limiting" \
    "for i in {1..20}; do curl -s -o /dev/null -w '%{http_code}\n' '$API_URL/health'; done | grep -q '429'" \
    "Verify that API implements rate limiting" \
    "medium"
  
  run_test "JWT Token Validation" \
    "curl -s -H 'Authorization: Bearer invalid.jwt.token' '$API_URL/protected-resource' | grep -q 'Invalid token'" \
    "Verify that invalid JWT tokens are rejected" \
    "high"
  
  run_test "CSRF Protection" \
    "curl -s -X POST '$API_URL/auth/login' -H 'Content-Type: application/json' -d '{\"username\":\"test\",\"password\":\"test\"}' | grep -q 'csrf'" \
    "Verify that CSRF tokens are provided" \
    "high"
  
  # 8. More intrusive tests if not skipped
  if [[ "$SKIP_INTRUSIVE" == "false" ]]; then
    echo -e "${YELLOW}Running potentially intrusive tests...${NC}"
    
    # Nikto web vulnerability scan
    if command -v nikto &> /dev/null; then
      echo -e "${YELLOW}Running Nikto web vulnerability scan...${NC}"
      echo "Nikto Web Vulnerability Scan" >> "$REPORT_FILE"
      
      local nikto_result=$(nikto -h "$API_URL" -Format txt)
      echo "$nikto_result" >> "$REPORT_FILE"
      
      # Display summary
      echo -e "${BLUE}Nikto scan summary:${NC}"
      echo "$nikto_result" | grep -E "^-"
    fi
    
    # Run directory brute force scan
    if command -v dirb &> /dev/null; then
      echo -e "${YELLOW}Running directory brute force scan...${NC}"
      echo "Directory Brute Force Scan" >> "$REPORT_FILE"
      
      local dirb_result=$(dirb "$API_URL" /usr/share/dirb/wordlists/common.txt -S)
      echo "$dirb_result" >> "$REPORT_FILE"
      
      # Display found directories
      echo -e "${BLUE}Directories found:${NC}"
      echo "$dirb_result" | grep "DIRECTORY"
    fi
  fi
}

# Run the security audit
run_security_audit

# Print summary
echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}Security Audit Summary${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo -e "Total Tests: ${TOTAL}"
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Warnings: ${YELLOW}${WARNING}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"

# Add summary to report
echo "Security Audit Summary" >> "$REPORT_FILE"
echo "====================" >> "$REPORT_FILE"
echo "Total Tests: $TOTAL" >> "$REPORT_FILE"
echo "Passed: $PASSED" >> "$REPORT_FILE"
echo "Warnings: $WARNING" >> "$REPORT_FILE"
echo "Failed: $FAILED" >> "$REPORT_FILE"

echo -e "${YELLOW}Complete security audit report saved to: ${REPORT_FILE}${NC}"

# Exit with status based on results
if [[ $FAILED -gt 0 ]]; then
  exit 1
else
  exit 0
fi 