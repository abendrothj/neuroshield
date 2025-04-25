/**
 * Test script to verify the integration between AI and blockchain
 * This script simulates sending AI threat detections to the blockchain
 */

require('dotenv').config();
const axios = require('axios');
const { Gateway, Wallets } = require('fabric-network');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

// Configuration
const API_URL = process.env.API_URL || 'http://localhost:3000';
const CONNECTION_PROFILE_PATH = process.env.CONNECTION_PROFILE_PATH || path.resolve(__dirname, './connection-profile.json');
const WALLET_PATH = process.env.WALLET_PATH || path.join(__dirname, './wallet');
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'neurashield-channel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'neurashield';
const IDENTITY = process.env.BLOCKCHAIN_IDENTITY || 'admin';

// Sample threat data (simulating AI detections)
const sampleThreats = [
  {
    threat_type: "DDoS",
    confidence: 0.92,
    raw_predictions: [0.05, 0.92, 0.01, 0.01, 0.01],
    source_data: {
      "packet_count": 2500,
      "byte_count": 1500000,
      "protocol": "TCP",
      "src_port": 80,
      "dst_port": 443,
      "tcp_flags": 16
    },
    timestamp: Date.now(),
    model_version: "v1.0.3"
  },
  {
    threat_type: "Brute Force",
    confidence: 0.87,
    raw_predictions: [0.08, 0.02, 0.87, 0.02, 0.01],
    source_data: {
      "packet_count": 350,
      "byte_count": 42000,
      "protocol": "TCP",
      "src_port": 22,
      "dst_port": 22,
      "tcp_flags": 24
    },
    timestamp: Date.now(),
    model_version: "v1.0.3"
  },
  {
    threat_type: "Port Scan",
    confidence: 0.78,
    raw_predictions: [0.15, 0.01, 0.05, 0.78, 0.01],
    source_data: {
      "packet_count": 120,
      "byte_count": 7200,
      "protocol": "TCP",
      "src_port": 43210,
      "dst_port": 1000,
      "tcp_flags": 2
    },
    timestamp: Date.now(),
    model_version: "v1.0.3"
  }
];

/**
 * Send a test threat to the AI detection webhook
 */
async function sendTestThreatToWebhook(threat) {
  try {
    console.log(`Sending threat (${threat.threat_type}) to webhook...`);
    const response = await axios.post(`${API_URL}/api/v1/ai-detection`, threat, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    console.log(`Response: ${response.status} - ${JSON.stringify(response.data)}`);
    return response.data;
  } catch (error) {
    console.error(`Error sending to webhook: ${error.message}`);
    if (error.response) {
      console.error(`Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}`);
    }
    throw error;
  }
}

/**
 * Query blockchain for events and verify if our test threat was logged
 */
async function verifyBlockchainEvents() {
  let gateway;
  
  try {
    console.log('Connecting to blockchain to verify events...');
    
    // Load connection profile
    const connectionProfile = JSON.parse(fs.readFileSync(CONNECTION_PROFILE_PATH, 'utf8'));
    
    // Create a new file system wallet for managing identities
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    
    // Check if admin identity exists in the wallet
    const identity = await wallet.get(IDENTITY);
    if (!identity) {
      throw new Error(`Identity ${IDENTITY} not found in the wallet. Please run enroll-admin.js first.`);
    }
    
    // Create a new gateway for connecting to the peer node
    gateway = new Gateway();
    await gateway.connect(connectionProfile, {
      wallet,
      identity: IDENTITY,
      discovery: { enabled: false }
    });
    
    // Get the network and contract
    const network = await gateway.getNetwork(CHANNEL_NAME);
    const contract = network.getContract(CHAINCODE_NAME);
    
    // Query all events
    console.log('Querying all events from blockchain...');
    const result = await contract.evaluateTransaction('QueryAllEvents');
    const events = JSON.parse(result.toString());
    
    console.log(`Found ${events.length} events in the blockchain`);
    
    // Display the most recent events (up to 5)
    const recentEvents = events.slice(-5);
    console.log('Recent events:');
    recentEvents.forEach((event, index) => {
      console.log(`Event ${index + 1}:`);
      console.log(`  ID: ${event.ID}`);
      console.log(`  Type: ${event.Type}`);
      console.log(`  Timestamp: ${event.Timestamp}`);
      console.log(`  Details: ${event.Details}`);
      console.log(`  IPFS Hash: ${event.IPFSHash || 'None'}`);
      console.log('');
    });
    
    return events;
  } catch (error) {
    console.error(`Error verifying blockchain events: ${error.message}`);
    throw error;
  } finally {
    // Disconnect from the gateway
    if (gateway) {
      gateway.disconnect();
    }
  }
}

/**
 * Main test function
 */
async function runIntegrationTest() {
  try {
    console.log('Starting AI-Blockchain integration test...');
    
    // Send test threats to webhook
    for (const threat of sampleThreats) {
      await sendTestThreatToWebhook(threat);
      // Wait a bit between requests
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // Wait for blockchain processing
    console.log('Waiting for blockchain processing (10 seconds)...');
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Verify events were stored in blockchain
    await verifyBlockchainEvents();
    
    console.log('Integration test completed successfully!');
    
  } catch (error) {
    console.error(`Integration test failed: ${error.message}`);
    process.exit(1);
  }
}

// Run the test
runIntegrationTest(); 