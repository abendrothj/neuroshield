/**
 * NeuraShield Simplified Blockchain Integration Test Script
 * 
 * This script tests the basic connectivity and operations with the Hyperledger Fabric blockchain
 * using the existing admin identity in the wallet.
 */

const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Configuration
const CONNECTION_PROFILE_PATH = process.env.CONNECTION_PROFILE_PATH || path.resolve(__dirname, './connection-profile.json');
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'neurashield-channel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'neurashield';
const WALLET_PATH = process.env.WALLET_PATH || path.join(__dirname, './wallet');
const IDENTITY = 'admin';

async function main() {
  try {
    console.log('Starting simplified blockchain integration test');
    
    // Load the connection profile
    const connectionProfile = JSON.parse(fs.readFileSync(CONNECTION_PROFILE_PATH, 'utf8'));
    console.log('Connection profile loaded successfully');
    
    // Get the wallet
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    console.log(`Wallet path: ${WALLET_PATH}`);
    
    // Check if admin identity exists
    const identity = await wallet.get(IDENTITY);
    if (!identity) {
      console.log(`An identity for the user "${IDENTITY}" does not exist in the wallet`);
      console.log('Run enrollAdmin.js first');
      return;
    }
    
    // Create a new gateway for connecting to the peer node
    const gateway = new Gateway();
    console.log('Connecting to gateway...');
    
    await gateway.connect(connectionProfile, {
      wallet,
      identity: IDENTITY,
      discovery: { enabled: false }
    });
    
    console.log('Connected to gateway successfully');
    
    // Get the network channel
    const network = await gateway.getNetwork(CHANNEL_NAME);
    console.log(`Connected to channel: ${CHANNEL_NAME}`);
    
    // Get the contract
    const contract = network.getContract(CHAINCODE_NAME);
    console.log(`Obtained contract: ${CHAINCODE_NAME}`);
    
    // Test 1: Query all events (read operation)
    console.log('\nTest 1: Querying all events...');
    try {
      const allEventsBuffer = await contract.evaluateTransaction('QueryAllEvents');
      const allEvents = JSON.parse(allEventsBuffer.toString());
      console.log(`Query successful. Found ${allEvents.length} events.`);
      if (allEvents.length > 0) {
        console.log('Sample event:', JSON.stringify(allEvents[0], null, 2));
      }
    } catch (error) {
      console.error(`Failed to query all events: ${error}`);
    }
    
    // Test 2: Submit a new event (write operation)
    console.log('\nTest 2: Submitting new event...');
    try {
      const eventId = `test-event-${crypto.randomBytes(4).toString('hex')}-${Date.now()}`;
      const timestamp = new Date().toISOString();
      const eventType = 'Test';
      const details = JSON.stringify({
        test_message: 'This is a test event from simple-test.js',
        timestamp: timestamp
      });
      const ipfsHash = '';
      
      console.log(`Submitting event with ID: ${eventId}`);
      await contract.submitTransaction(
        'LogEvent',
        eventId,
        timestamp,
        eventType,
        details,
        ipfsHash
      );
      console.log('Event submitted successfully');
      
      // Verify event was added
      const eventBuffer = await contract.evaluateTransaction('QueryEvent', eventId);
      const event = JSON.parse(eventBuffer.toString());
      console.log('Retrieved event:', JSON.stringify(event, null, 2));
    } catch (error) {
      console.error(`Failed to submit event: ${error}`);
    }
    
    // Disconnect from the gateway
    gateway.disconnect();
    console.log('Disconnected from gateway');
    
    console.log('\nIntegration test completed');
  } catch (error) {
    console.error(`Failed to run test: ${error}`);
    console.error(error.stack);
    process.exit(1);
  }
}

main(); 