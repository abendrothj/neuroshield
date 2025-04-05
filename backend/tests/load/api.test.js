const axios = require('axios');

// Configuration
const API_URL = 'http://localhost:3001';
const AI_URL = 'http://localhost:5000';
const CONCURRENCY = 10;
const REQUESTS_PER_ENDPOINT = 20;
const MAX_RESPONSE_TIME = 1000; // 1000ms threshold

// Test endpoints
const ENDPOINTS = [
  { url: `${API_URL}/health`, method: 'GET' },
  { url: `${API_URL}/api/metrics`, method: 'GET' },
  { url: `${API_URL}/api/events`, method: 'GET' },
  { url: `${AI_URL}/health`, method: 'GET' },
  { 
    url: `${AI_URL}/analyze`, 
    method: 'POST',
    data: { 
      data: [createTestSample()]
    }
  }
];

// Create a sample with the required 39 features
function createTestSample() {
  const sample = {};
  for (let i = 0; i < 39; i++) {
    sample[`feature_${i}`] = Math.random();
  }
  return sample;
}

// Sleep function for rate limiting
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Make a request to a specific endpoint and measure response time
async function makeRequest(endpoint, requestId) {
  const startTime = Date.now();
  
  try {
    let response;
    
    if (endpoint.method === 'GET') {
      response = await axios.get(endpoint.url);
    } else if (endpoint.method === 'POST') {
      response = await axios.post(endpoint.url, endpoint.data || {});
    }
    
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    return {
      endpoint: endpoint.url,
      method: endpoint.method,
      requestId,
      responseTime,
      status: response.status,
      success: true
    };
  } catch (error) {
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    return {
      endpoint: endpoint.url,
      method: endpoint.method,
      requestId,
      responseTime,
      status: error.response ? error.response.status : 0,
      success: false,
      error: error.message
    };
  }
}

// Run load test for a specific endpoint
async function testEndpoint(endpoint, concurrency, totalRequests) {
  console.log(`\nTesting endpoint: ${endpoint.url} (${endpoint.method})`);
  console.log(`Concurrency: ${concurrency}, Total Requests: ${totalRequests}`);
  
  // Generate all requests but don't await them yet
  const requests = [];
  for (let i = 0; i < totalRequests; i++) {
    requests.push(makeRequest(endpoint, i + 1));
    
    // Add a small delay between creating requests to avoid rate limiting
    if (i % concurrency === 0) {
      await sleep(50);
    }
  }
  
  // Wait for all requests to complete
  const results = await Promise.all(requests);
  
  // Calculate statistics
  const responseTimes = results.map(r => r.responseTime);
  const successCount = results.filter(r => r.success).length;
  const failureCount = results.filter(r => !r.success).length;
  
  const min = Math.min(...responseTimes);
  const max = Math.max(...responseTimes);
  const avg = responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
  
  // Sort response times and get the 95th percentile
  const sortedTimes = [...responseTimes].sort((a, b) => a - b);
  const p95Index = Math.floor(sortedTimes.length * 0.95);
  const p95 = sortedTimes[p95Index];
  
  // Print results
  console.log(`\nResults for ${endpoint.url} (${endpoint.method}):`);
  console.log(`Success: ${successCount}/${totalRequests} (${((successCount/totalRequests)*100).toFixed(2)}%)`);
  console.log(`Response times (ms):`);
  console.log(`  Min: ${min}`);
  console.log(`  Max: ${max}`);
  console.log(`  Avg: ${avg.toFixed(2)}`);
  console.log(`  95th percentile: ${p95}`);
  
  // Determine if the endpoint meets performance criteria
  const performanceMet = p95 < MAX_RESPONSE_TIME;
  console.log(performanceMet 
    ? `✅ PASSED: 95th percentile (${p95}ms) is below threshold (${MAX_RESPONSE_TIME}ms)`
    : `❌ FAILED: 95th percentile (${p95}ms) exceeds threshold (${MAX_RESPONSE_TIME}ms)`);
  
  return {
    endpoint: endpoint.url,
    method: endpoint.method,
    successRate: successCount / totalRequests,
    min,
    max,
    avg,
    p95,
    performanceMet
  };
}

// Run load tests for all endpoints
async function runLoadTests() {
  console.log('API Load Testing');
  console.log('===============');
  
  const results = [];
  
  for (const endpoint of ENDPOINTS) {
    const result = await testEndpoint(endpoint, CONCURRENCY, REQUESTS_PER_ENDPOINT);
    results.push(result);
  }
  
  // Print summary
  console.log('\nLoad Test Summary:');
  console.log('================');
  console.log('Endpoint | Method | Success Rate | Avg (ms) | 95th (ms) | Result');
  console.log('---------|--------|--------------|----------|-----------|-------');
  
  results.forEach(result => {
    const endpointPath = new URL(result.endpoint).pathname;
    console.log(
      `${endpointPath.padEnd(9)} | ` +
      `${result.method.padEnd(6)} | ` +
      `${(result.successRate * 100).toFixed(2)}%.padEnd(12) | ` +
      `${result.avg.toFixed(2).padEnd(8)} | ` +
      `${result.p95.toString().padEnd(9)} | ` +
      `${result.performanceMet ? '✅ PASS' : '❌ FAIL'}`
    );
  });
  
  // Determine overall result
  const overallSuccess = results.every(r => r.performanceMet);
  console.log(`\nOverall Result: ${overallSuccess ? '✅ PASSED' : '❌ FAILED'}`);
  
  return overallSuccess;
}

// Run the test
async function runTest() {
  try {
    const success = await runLoadTests();
    process.exit(success ? 0 : 1);
  } catch (error) {
    console.error('Unhandled error during load testing:', error);
    process.exit(1);
  }
}

// If this file is run directly, execute the test
if (require.main === module) {
  runTest();
}

module.exports = { runLoadTests, runTest }; 