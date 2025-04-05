const axios = require('axios');

async function testIntegration() {
  try {
    console.log('Testing NeuraShield backend integration with AI service...');
    
    // Generate test data with 39 features
    const features = Array(39).fill(0).map(() => Math.random());
    
    // Create sample with feature names
    const sample = {};
    for (let i = 0; i < features.length; i++) {
      sample[`feature_${i}`] = features[i];
    }
    
    // Format data as expected by the API
    const data = {
      data: [sample]  // The API expects an array of objects
    };
    
    console.log('Sending test data to backend analyze endpoint...');
    console.log('Request payload:', JSON.stringify(data, null, 2));
    
    const response = await axios.post('http://localhost:3001/api/analyze', data);
    
    console.log('\nResponse from backend:');
    console.log(JSON.stringify(response.data, null, 2));
    
    console.log('\nTesting AI metrics endpoint...');
    const metricsResponse = await axios.get('http://localhost:3001/api/ai-metrics');
    
    console.log('\nMetrics response:');
    console.log(JSON.stringify(metricsResponse.data, null, 2));
    
    console.log('\nIntegration test completed successfully!');
  } catch (error) {
    console.error('Error during integration test:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
    }
  }
}

testIntegration(); 