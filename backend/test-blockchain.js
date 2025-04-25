/**
 * NeuraShield Blockchain Integration Test Script
 * 
 * This script tests the integration between NeuraShield and the Hyperledger Fabric blockchain.
 * It verifies identity management, event submission, and retrieval functionality.
 */

const identityManager = require('./identity-manager');
const blockchainIntegration = require('./src/blockchain-integration');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

// Configure test settings from environment or defaults
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'neurashield-channel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'neurashield';
const TEST_IDENTITY = 'test-user-' + Date.now();
const TEST_EVENT_COUNT = 3;

// Initialize logger
const winston = require('winston');
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.colorize(),
    winston.format.timestamp(),
    winston.format.printf(({ level, message, timestamp }) => {
      return `${timestamp} ${level}: ${message}`;
    })
  ),
  transports: [
    new winston.transports.Console()
  ]
});

/**
 * Main test function
 */
async function runBlockchainTests() {
  logger.info('Starting NeuraShield Blockchain Integration Tests');
  
  try {
    // Step 1: Initialize the blockchain
    logger.info('Step 1: Initializing blockchain connection...');
    const initialized = await blockchainIntegration.initBlockchain();
    if (!initialized) {
      throw new Error('Blockchain initialization failed');
    }
    logger.info('✅ Blockchain initialized successfully');

    // Step 2: Test identity management
    logger.info('Step 2: Testing identity management...');
    await testIdentityManagement();
    logger.info('✅ Identity management tests passed');

    // Step 3: Test event submission
    logger.info('Step 3: Testing event submission...');
    const eventIds = await testEventSubmission();
    logger.info('✅ Event submission tests passed');

    // Step 4: Test event retrieval
    logger.info('Step 4: Testing event retrieval...');
    await testEventRetrieval(eventIds);
    logger.info('✅ Event retrieval tests passed');

    // Step 5: Check blockchain status
    logger.info('Step 5: Verifying blockchain status...');
    await testBlockchainStatus();
    logger.info('✅ Blockchain status verified');

    logger.info('🎉 All blockchain integration tests passed!');
    process.exit(0);
  } catch (error) {
    logger.error(`❌ Test failed: ${error.message}`);
    if (error.stack) {
      logger.error(error.stack);
    }
    process.exit(1);
  }
}

/**
 * Test identity management functionality
 */
async function testIdentityManagement() {
  // Test admin identity
  const adminValid = await identityManager.verifyIdentity('admin');
  if (!adminValid) {
    throw new Error('Admin identity verification failed');
  }
  logger.info('Admin identity verified successfully');

  // Test user registration (if not in production)
  if (process.env.NODE_ENV !== 'production') {
    try {
      // Register a test user
      await identityManager.registerUser(
        TEST_IDENTITY, 
        'client',
        'org1.department1'
      );
      logger.info(`Test user ${TEST_IDENTITY} registered successfully`);
      
      // Verify the test user identity
      const userValid = await identityManager.verifyIdentity(TEST_IDENTITY);
      if (!userValid) {
        throw new Error(`Test user ${TEST_IDENTITY} verification failed`);
      }
      logger.info(`Test user ${TEST_IDENTITY} verified successfully`);
      
      // List all identities for verification
      const identities = await identityManager.listIdentities();
      logger.info(`Found ${identities.length} identities in wallet`);
      
      // Clean up test user after verification
      if (process.env.KEEP_TEST_IDENTITY !== 'true') {
        await identityManager.deleteIdentity(TEST_IDENTITY);
        logger.info(`Test user ${TEST_IDENTITY} cleaned up`);
      }
    } catch (error) {
      throw new Error(`Identity management test failed: ${error.message}`);
    }
  } else {
    logger.info('Skipping test user registration in production mode');
  }
}

/**
 * Test security event submission
 */
async function testEventSubmission() {
  const eventIds = [];
  
  // Generate and submit test events
  for (let i = 0; i < TEST_EVENT_COUNT; i++) {
    // Create test threat data
    const threatData = {
      threat_type: `test-threat-type-${i}`,
      confidence: Math.random() * 0.9 + 0.1, // Random confidence between 0.1 and 1.0
      source_ip: `192.168.1.${Math.floor(Math.random() * 255)}`,
      timestamp: new Date().toISOString(),
      target_resource: 'test-resource',
      metadata: {
        test_id: `test-${Date.now()}-${i}`,
        environment: 'test'
      }
    };
    
    // Process the event
    logger.info(`Submitting test event ${i+1}/${TEST_EVENT_COUNT}...`);
    const result = await blockchainIntegration.processSecurityEvent(threatData);
    
    if (!result || !result.eventId) {
      throw new Error(`Event ${i+1}/${TEST_EVENT_COUNT} submission failed`);
    }
    
    logger.info(`Event ${i+1}/${TEST_EVENT_COUNT} submitted successfully with ID: ${result.eventId}`);
    if (result.ipfsHash && result.ipfsHash !== '') {
      logger.info(`Event stored in IPFS with hash: ${result.ipfsHash}`);
    }
    
    eventIds.push(result.eventId);
  }
  
  // Allow time for transactions to be committed
  logger.info('Waiting for transactions to be committed...');
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  return eventIds;
}

/**
 * Test event retrieval from the blockchain
 */
async function testEventRetrieval(eventIds) {
  // Get a gateway connection
  const gateway = await identityManager.getGatewayConnection('admin');
  const network = await gateway.getNetwork(CHANNEL_NAME);
  const contract = network.getContract(CHAINCODE_NAME);
  
  // Test retrieving individual events
  for (const eventId of eventIds) {
    try {
      logger.info(`Retrieving event with ID: ${eventId}`);
      const resultBuffer = await contract.evaluateTransaction('QueryEvent', eventId);
      const event = JSON.parse(resultBuffer.toString());
      
      if (!event || event.ID !== eventId) {
        throw new Error(`Failed to retrieve event with ID: ${eventId}`);
      }
      
      logger.info(`Successfully retrieved event: ${JSON.stringify(event, null, 2)}`);
    } catch (error) {
      throw new Error(`Event retrieval test failed for ID ${eventId}: ${error.message}`);
    }
  }
  
  // Test retrieving all events
  try {
    logger.info('Retrieving all events...');
    const resultBuffer = await contract.evaluateTransaction('QueryAllEvents');
    const events = JSON.parse(resultBuffer.toString());
    
    logger.info(`Retrieved ${events.length} events from blockchain`);
    
    // Verify our test events are included
    for (const eventId of eventIds) {
      const found = events.some(e => e.ID === eventId);
      if (!found) {
        throw new Error(`Event with ID ${eventId} not found in all events query`);
      }
    }
    
    logger.info('All test events verified in the blockchain');
  } catch (error) {
    throw new Error(`All events retrieval test failed: ${error.message}`);
  } finally {
    gateway.disconnect();
  }
}

/**
 * Test blockchain status checking
 */
async function testBlockchainStatus() {
  // This simulates what the server does to check blockchain status
  let gateway;
  try {
    gateway = await identityManager.getGatewayConnection('admin');
    const network = await gateway.getNetwork(CHANNEL_NAME);
    
    if (!network) {
      throw new Error('Failed to connect to network');
    }
    
    logger.info('Connected to blockchain network successfully');
    
    // Test contract access
    const contract = network.getContract(CHAINCODE_NAME);
    await contract.evaluateTransaction('QueryAllEvents');
    
    logger.info('Successfully queried chaincode');
    return true;
  } catch (error) {
    throw new Error(`Blockchain status check failed: ${error.message}`);
  } finally {
    if (gateway) {
      gateway.disconnect();
    }
  }
}

// Run the tests
runBlockchainTests(); 