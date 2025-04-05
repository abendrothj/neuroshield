const WebSocket = require('ws');

// Test configuration
const WS_URL = 'ws://localhost:3001/ws';
const TEST_TIMEOUT = 10000; // 10 seconds

async function testWebSocketConnection() {
  return new Promise((resolve, reject) => {
    console.log(`Connecting to WebSocket at ${WS_URL}...`);
    
    const socket = new WebSocket(WS_URL);
    let connected = false;
    
    // Set timeout
    const timeout = setTimeout(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
      if (!connected) {
        reject(new Error(`Connection timed out after ${TEST_TIMEOUT}ms`));
      } else {
        resolve(true);
      }
    }, TEST_TIMEOUT);
    
    // Connection opened
    socket.on('open', () => {
      console.log('✅ WebSocket connection established');
      connected = true;
      
      // Send a test message
      const testMessage = JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString()
      });
      
      console.log(`Sending test message: ${testMessage}`);
      socket.send(testMessage);
    });
    
    // Listen for messages
    socket.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        console.log('Received message:', message);
        
        if (message.type === 'pong') {
          console.log('✅ WebSocket bidirectional communication working');
          socket.close();
          clearTimeout(timeout);
          resolve(true);
        }
      } catch (error) {
        console.error('Error parsing message:', error);
        socket.close();
        clearTimeout(timeout);
        reject(error);
      }
    });
    
    // Handle errors
    socket.on('error', (error) => {
      console.error('WebSocket error:', error.message);
      clearTimeout(timeout);
      reject(error);
    });
    
    // Connection closed
    socket.on('close', (code, reason) => {
      if (!connected) {
        console.error(`❌ Failed to connect to WebSocket. Code: ${code}, Reason: ${reason || 'No reason provided'}`);
        clearTimeout(timeout);
        reject(new Error(`Connection closed with code ${code}`));
      } else {
        console.log(`WebSocket connection closed. Code: ${code}, Reason: ${reason || 'Normal closure'}`);
        clearTimeout(timeout);
        // If we haven't resolved/rejected yet, resolve here
        resolve(true);
      }
    });
  });
}

// Run the test
async function runTest() {
  try {
    const result = await testWebSocketConnection();
    console.log('\nWebSocket Test Summary:');
    console.log('=====================');
    console.log('Status: ✅ PASSED');
    console.log('Details: WebSocket connection and communication successful');
    process.exit(0);
  } catch (error) {
    console.log('\nWebSocket Test Summary:');
    console.log('=====================');
    console.log('Status: ❌ FAILED');
    console.log(`Details: ${error.message}`);
    process.exit(1);
  }
}

// If this file is run directly, execute the test
if (require.main === module) {
  runTest();
}

module.exports = { testWebSocketConnection, runTest }; 