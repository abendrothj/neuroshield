const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Configuration
const AI_SERVICE_URL = 'http://localhost:5000';
const TEST_ITERATIONS = 5;
const BATCH_SIZES = [1, 10, 50, 100];
const PERFORMANCE_THRESHOLD = 1000; // 1000ms for batch processing

// Create a sample with the required 39 features
function createSample(seed = 0) {
  const sample = {};
  for (let i = 0; i < 39; i++) {
    // Create deterministic but varied values based on seed and feature index
    sample[`feature_${i}`] = Math.abs(Math.sin(i * 0.1 + seed)) * 0.9;
  }
  return sample;
}

// Create a batch of test samples
function createBatch(batchSize, startSeed = 0) {
  const batch = [];
  for (let i = 0; i < batchSize; i++) {
    batch.push(createSample(startSeed + i));
  }
  return batch;
}

// Test the AI model's performance
async function testModelPerformance() {
  console.log('Testing AI Model Performance');
  console.log('============================');
  
  try {
    // First check if the AI service is available
    const healthResponse = await axios.get(`${AI_SERVICE_URL}/health`);
    console.log(`AI Service Health: ${healthResponse.data.status}`);
    console.log(`Model Version: ${healthResponse.data.model_version}`);
    console.log(`GPU Available: ${healthResponse.data.gpu_available}`);
    
    // Results collection
    const results = [];
    
    // Test with different batch sizes
    for (const batchSize of BATCH_SIZES) {
      console.log(`\nTesting with batch size: ${batchSize}`);
      
      const batchTimes = [];
      let avgTimePerSample = 0;
      
      // Run multiple iterations for more stable results
      for (let i = 0; i < TEST_ITERATIONS; i++) {
        const testBatch = createBatch(batchSize, i * batchSize);
        
        const payload = {
          data: testBatch
        };
        
        const startTime = Date.now();
        const response = await axios.post(`${AI_SERVICE_URL}/analyze`, payload);
        const endTime = Date.now();
        
        const totalTime = endTime - startTime;
        const processingTime = response.data.processing_time * 1000; // Convert to ms
        const timePerSample = processingTime / batchSize;
        
        batchTimes.push(totalTime);
        avgTimePerSample += timePerSample;
        
        console.log(`  Iteration ${i+1}/${TEST_ITERATIONS}: ${totalTime}ms (${timePerSample.toFixed(2)}ms/sample)`);
      }
      
      // Calculate averages
      const avgBatchTime = batchTimes.reduce((sum, time) => sum + time, 0) / TEST_ITERATIONS;
      avgTimePerSample = avgTimePerSample / TEST_ITERATIONS;
      
      console.log(`  Average time for batch size ${batchSize}: ${avgBatchTime.toFixed(2)}ms`);
      console.log(`  Average time per sample: ${avgTimePerSample.toFixed(2)}ms`);
      
      results.push({
        batchSize,
        avgBatchTime,
        avgTimePerSample,
        iterations: TEST_ITERATIONS
      });
    }
    
    // Generate summary
    console.log('\nPerformance Test Summary:');
    console.log('========================');
    console.log('Batch Size | Total Time (ms) | Time per Sample (ms)');
    console.log('----------|-----------------|--------------------');
    
    results.forEach(result => {
      console.log(`${result.batchSize.toString().padEnd(10)} | ${result.avgBatchTime.toFixed(2).padEnd(15)} | ${result.avgTimePerSample.toFixed(2)}`);
    });
    
    // Check against performance threshold
    const largestBatchResult = results.find(r => r.batchSize === Math.max(...BATCH_SIZES));
    if (largestBatchResult && largestBatchResult.avgBatchTime < PERFORMANCE_THRESHOLD) {
      console.log(`\n✅ PASSED: Model performance meets threshold (${largestBatchResult.avgBatchTime.toFixed(2)}ms < ${PERFORMANCE_THRESHOLD}ms for batch size ${largestBatchResult.batchSize})`);
      return true;
    } else if (largestBatchResult) {
      console.log(`\n❌ FAILED: Model performance below threshold (${largestBatchResult.avgBatchTime.toFixed(2)}ms > ${PERFORMANCE_THRESHOLD}ms for batch size ${largestBatchResult.batchSize})`);
      return false;
    } else {
      console.log('\n❌ FAILED: Could not determine performance for largest batch size');
      return false;
    }
    
  } catch (error) {
    console.error('Error during performance testing:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
    }
    return false;
  }
}

// Run the test
async function runTest() {
  const success = await testModelPerformance();
  process.exit(success ? 0 : 1);
}

// If this file is run directly, execute the test
if (require.main === module) {
  runTest();
}

module.exports = { testModelPerformance, runTest }; 