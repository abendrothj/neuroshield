const axios = require('axios');

async function testDirectAPI() {
  try {
    console.log('Testing direct connection to AI service...');
    
    // Check health endpoint first
    console.log('Checking AI service health...');
    const healthResponse = await axios.get('http://localhost:5000/health');
    console.log('Health response:', healthResponse.data);
    
    // Generate test data with 39 features
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

testDirectAPI(); 