const axios = require('axios');
const path = require('path');
const fs = require('fs');

// Load test data from fixtures
const loadTestFixture = (filename) => {
  const fixturePath = path.join(__dirname, 'fixtures', filename);
  try {
    const data = fs.readFileSync(fixturePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error(`Error loading fixture ${filename}:`, error.message);
    return null;
  }
};

async function testDirectAPI() {
  try {
    console.log('Testing direct connection to AI service...');
    
    // Check health endpoint first
    console.log('Checking AI service health...');
    const healthResponse = await axios.get('http://localhost:5000/health');
    console.log('Health response:', healthResponse.data);
    
    // Try to load test data from fixtures
    const testPayload = loadTestFixture('test_payload.json');
    
    // If fixture was loaded successfully, use it
    if (testPayload) {
      try {
        console.log('\nUsing test payload from fixture:');
        console.log(JSON.stringify(testPayload, null, 2));
        
        const response = await axios.post(
          'http://localhost:5000/analyze', 
          testPayload
        );
        
        console.log('\nSuccess! Response:');
        console.log(JSON.stringify(response.data, null, 2));
        
        console.log('\n✅ Test successful with fixture data.');
        return testPayload;
      } catch (error) {
        console.error(`❌ Error with fixture payload:`, error.message);
        if (error.response) {
          console.error('Status:', error.response.status);
          console.error('Response:', error.response.data);
        }
      }
    }
    
    // Generate test data with 39 features as fallback
    const features = Array(39).fill(0).map(() => Math.random());
    
    // Create sample with feature names
    const sample = {};
    for (let i = 0; i < features.length; i++) {
      sample[`feature_${i}`] = features[i];
    }
    
    // Try different request formats
    const testCases = [
      { name: "Single object", payload: { data: sample } },
      { name: "Array of objects", payload: { data: [sample] } }
    ];
    
    for (const testCase of testCases) {
      try {
        console.log(`\nTrying ${testCase.name}:`);
        console.log(JSON.stringify(testCase.payload, null, 2));
        
        const response = await axios.post(
          'http://localhost:5000/analyze', 
          testCase.payload
        );
        
        console.log('\nSuccess! Response:');
        console.log(JSON.stringify(response.data, null, 2));
        
        // If we get here, we found the right format
        console.log('\n✅ This is the correct format to use.');
        return testCase.payload;
      } catch (error) {
        console.error(`❌ Error with ${testCase.name}:`, error.message);
        if (error.response) {
          console.error('Status:', error.response.status);
          console.error('Response:', error.response.data);
        }
      }
    }
    
    console.error('\nAll test cases failed.');
    
  } catch (error) {
    console.error('Error during API test:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
    }
  }
}

// If this file is run directly, execute the test
if (require.main === module) {
  testDirectAPI();
}

// Export the function for use in other tests
module.exports = { testDirectAPI }; 