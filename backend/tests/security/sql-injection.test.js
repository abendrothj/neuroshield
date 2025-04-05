const axios = require('axios');

// Configuration
const API_URL = 'http://localhost:3001';
const ENDPOINTS = [
  '/api/events',
  '/api/users',
  '/api/alerts',
  '/api/search'
];

// SQL Injection payloads to test
const PAYLOADS = [
  "' OR '1'='1",
  "'; DROP TABLE users; --",
  "' UNION SELECT * FROM users; --",
  "' OR 1=1; --",
  "admin'; --",
  "' OR ''='",
  "1' OR '1' = '1'\"))",
  "; SELECT * FROM information_schema.tables",
  "' OR 1=1 #",
  "' OR 'x'='x",
  "1' OR '1'='1' /*",
  "'' OR 1 -- -",
  "' UNION SELECT 1,@@version,3,4,5; --"
];

// Test SQL Injection vulnerabilities
async function testSqlInjection() {
  console.log('Testing SQL Injection Vulnerabilities');
  console.log('====================================');
  
  let vulnerabilities = 0;
  let totalTests = 0;
  
  try {
    // Test GET endpoints with query parameters
    for (const endpoint of ENDPOINTS) {
      console.log(`\nTesting endpoint: ${endpoint} (GET)`);
      
      for (const payload of PAYLOADS) {
        totalTests++;
        
        try {
          // Encode the payload for URL parameters
          const encodedPayload = encodeURIComponent(payload);
          const url = `${API_URL}${endpoint}?q=${encodedPayload}`;
          
          console.log(`  Testing: ${url}`);
          
          const response = await axios.get(url);
          
          // Check if the response has signs of SQL injection success
          const responseData = JSON.stringify(response.data).toLowerCase();
          const suspiciousPatterns = [
            'sql', 'syntax', 'mysql', 'postgresql', 'sqlite', 
            'table', 'database', 'column', 'select', 'version()',
            'information_schema'
          ];
          
          const hasSuspiciousData = suspiciousPatterns.some(pattern => 
            responseData.includes(pattern)
          );
          
          if (hasSuspiciousData) {
            console.log(`  ❌ VULNERABLE: Endpoint ${endpoint} with payload: ${payload}`);
            console.log(`  Response contained suspicious data: ${responseData.substring(0, 100)}...`);
            vulnerabilities++;
          } else {
            console.log(`  ✅ Passed: Endpoint properly handled the payload`);
          }
        } catch (error) {
          // 4xx errors are good - they indicate input validation
          if (error.response && error.response.status >= 400 && error.response.status < 500) {
            console.log(`  ✅ Passed: Endpoint rejected the payload with status ${error.response.status}`);
          } else {
            console.log(`  ⚠️ Error: ${error.message}`);
          }
        }
      }
    }
    
    // Test POST endpoints with JSON payloads
    for (const endpoint of ENDPOINTS) {
      console.log(`\nTesting endpoint: ${endpoint} (POST)`);
      
      for (const payload of PAYLOADS) {
        totalTests++;
        
        try {
          const data = {
            query: payload,
            name: `test ${payload}`,
            id: payload
          };
          
          console.log(`  Testing payload in POST body: ${JSON.stringify(data)}`);
          
          const response = await axios.post(`${API_URL}${endpoint}`, data);
          
          // Check if the response has signs of SQL injection success
          const responseData = JSON.stringify(response.data).toLowerCase();
          const suspiciousPatterns = [
            'sql', 'syntax', 'mysql', 'postgresql', 'sqlite', 
            'table', 'database', 'column', 'select', 'version()',
            'information_schema'
          ];
          
          const hasSuspiciousData = suspiciousPatterns.some(pattern => 
            responseData.includes(pattern)
          );
          
          if (hasSuspiciousData) {
            console.log(`  ❌ VULNERABLE: Endpoint ${endpoint} with payload: ${payload}`);
            console.log(`  Response contained suspicious data: ${responseData.substring(0, 100)}...`);
            vulnerabilities++;
          } else {
            console.log(`  ✅ Passed: Endpoint properly handled the payload`);
          }
        } catch (error) {
          // 4xx errors are good - they indicate input validation
          if (error.response && error.response.status >= 400 && error.response.status < 500) {
            console.log(`  ✅ Passed: Endpoint rejected the payload with status ${error.response.status}`);
          } else {
            console.log(`  ⚠️ Error: ${error.message}`);
          }
        }
      }
    }
    
    // Print summary
    console.log('\nSQL Injection Test Summary:');
    console.log('==========================');
    console.log(`Total tests: ${totalTests}`);
    console.log(`Potential vulnerabilities: ${vulnerabilities}`);
    
    if (vulnerabilities === 0) {
      console.log('✅ PASSED: No SQL injection vulnerabilities detected');
      return true;
    } else {
      console.log(`❌ FAILED: Found ${vulnerabilities} potential SQL injection vulnerabilities`);
      return false;
    }
    
  } catch (error) {
    console.error('Error during SQL injection testing:', error.message);
    return false;
  }
}

// Run the test
async function runTest() {
  try {
    const success = await testSqlInjection();
    process.exit(success ? 0 : 1);
  } catch (error) {
    console.error('Unhandled error:', error);
    process.exit(1);
  }
}

// If this file is run directly, execute the test
if (require.main === module) {
  runTest();
}

module.exports = { testSqlInjection, runTest }; 