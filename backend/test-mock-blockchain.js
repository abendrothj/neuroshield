/**
 * NeuraShield Mock Blockchain Integration Test Script
 * 
 * This script tests the mock blockchain integration for NeuraShield.
 */

const mockBlockchain = require('./mock-blockchain');
const crypto = require('crypto');
const winston = require('winston');

// Configure test settings
const TEST_EVENT_COUNT = 5;

// Initialize logger
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
async function runMockBlockchainTests() {
  logger.info('Starting NeuraShield Mock Blockchain Integration Tests');
  
  try {
    // Step 1: Initialize the mock blockchain
    logger.info('Step 1: Initializing mock blockchain...');
    const initialized = await mockBlockchain.initBlockchain();
    if (!initialized) {
      throw new Error('Mock blockchain initialization failed');
    }
    logger.info('✅ Mock blockchain initialized successfully');

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

    logger.info('🎉 All mock blockchain integration tests passed!');
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
  const adminValid = await mockBlockchain.verifyIdentity('admin');
  if (!adminValid) {
    throw new Error('Admin identity verification failed');
  }
  logger.info('Admin identity verified successfully');

  // Test user registration
  const testIdentity = 'test-user-' + Date.now();
  const registered = await mockBlockchain.registerUser(
    testIdentity, 
    'client',
    'org1.department1'
  );
  
  if (!registered) {
    throw new Error(`Test user ${testIdentity} registration failed`);
  }
  logger.info(`Test user ${testIdentity} registered successfully`);
  
  // Verify the test user identity
  const userValid = await mockBlockchain.verifyIdentity(testIdentity);
  if (!userValid) {
    throw new Error(`Test user ${testIdentity} verification failed`);
  }
  logger.info(`Test user ${testIdentity} verified successfully`);
  
  // List all identities for verification
  const identities = await mockBlockchain.listIdentities();
  logger.info(`Found ${identities.length} identities in mock wallet`);
  if (!identities.includes(testIdentity)) {
    throw new Error(`Test user ${testIdentity} not found in identity list`);
  }
  
  // Clean up test user after verification
  const deleted = await mockBlockchain.deleteIdentity(testIdentity);
  if (!deleted) {
    throw new Error(`Failed to delete test user ${testIdentity}`);
  }
  logger.info(`Test user ${testIdentity} cleaned up`);
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
    const result = await mockBlockchain.processSecurityEvent(threatData);
    
    if (!result || !result.eventId) {
      throw new Error(`Event ${i+1}/${TEST_EVENT_COUNT} submission failed`);
    }
    
    logger.info(`Event ${i+1}/${TEST_EVENT_COUNT} submitted successfully with ID: ${result.eventId}`);
    if (result.ipfsHash && result.ipfsHash !== '') {
      logger.info(`Event stored in mock IPFS with hash: ${result.ipfsHash}`);
    }
    
    eventIds.push(result.eventId);
  }
  
  return eventIds;
}

/**
 * Test event retrieval from the mock ledger
 */
async function testEventRetrieval(eventIds) {
  // Get all events
  const allEvents = await mockBlockchain.fetchEvents();
  logger.info(`Retrieved ${allEvents.length} events from mock ledger`);
  
  // Verify the initialization event exists
  const initEvent = allEvents.find(e => e.ID === 'init1');
  if (!initEvent) {
    throw new Error('Initialization event not found');
  }
  logger.info('Initialization event found in mock ledger');
  
  // Check for all test events
  for (const eventId of eventIds) {
    const event = await mockBlockchain.fetchEvent(eventId);
    if (!event) {
      throw new Error(`Event ${eventId} not found in mock ledger`);
    }
    logger.info(`Successfully retrieved event: ${event.ID}`);
    
    // Verify event properties
    if (event.ID !== eventId) {
      throw new Error(`Event ID mismatch: expected ${eventId}, got ${event.ID}`);
    }
    
    if (!event.Timestamp || !event.Type || !event.Details) {
      throw new Error(`Event ${eventId} is missing required properties`);
    }
  }
  
  logger.info('All test events verified in the mock ledger');
}

// Run the tests
runMockBlockchainTests(); 