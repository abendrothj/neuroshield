/**
 * NeuraShield Blockchain Verification Test
 * 
 * This script tests the blockchain verification system for secure event logging
 */

const assert = require('assert');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');

// Load the blockchain adapter
const { getBlockchainImplementation } = require('../backend/src/blockchain-adapter');

// Test event data
const testEvent = {
  id: `test-event-${Date.now()}`,
  timestamp: new Date().toISOString(),
  type: 'TEST_EVENT',
  details: {
    test_field: 'Test value',
    severity: 'medium',
    source: 'verification_test'
  }
};

async function runTests() {
  console.log('Starting blockchain verification tests...');
  
  try {
    // Get blockchain implementation
    const blockchain = await getBlockchainImplementation();
    await blockchain.initBlockchain();
    
    // Test 1: Process security event
    console.log('\nTest 1: Processing security event');
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    
    assert(eventResult.success, 'Event processing failed');
    assert(eventResult.eventId, 'No event ID returned');
    assert(eventResult.txId, 'No transaction ID returned');
    assert(eventResult.eventHash, 'No event hash returned');
    
    console.log('✅ Event processed successfully');
    console.log(`   Event ID: ${eventResult.eventId}`);
    console.log(`   Transaction ID: ${eventResult.txId}`);
    console.log(`   Event Hash: ${eventResult.eventHash}`);
    
    // Test 2: Fetch event
    console.log('\nTest 2: Fetching event');
    const fetchedEvent = await blockchain.fetchEvent(eventResult.eventId);
    
    assert(fetchedEvent, 'Failed to fetch event');
    assert(fetchedEvent.id === eventResult.eventId, 'Event ID mismatch');
    assert(fetchedEvent.eventHash === eventResult.eventHash, 'Event hash mismatch');
    
    console.log('✅ Event fetched successfully');
    
    // Test 3: Verify event integrity
    console.log('\nTest 3: Verifying event integrity');
    const integrityResult = blockchain.verifyEventIntegrity(fetchedEvent);
    
    assert(integrityResult.isValid, 'Event integrity check failed');
    assert(integrityResult.calculatedHash === fetchedEvent.eventHash, 'Hash mismatch');
    
    console.log('✅ Event integrity verified');
    
    // Test 4: Verify transaction
    console.log('\nTest 4: Verifying transaction');
    const txVerification = await blockchain.verifyTransaction(fetchedEvent.txId);
    
    assert(txVerification.isValid, 'Transaction verification failed');
    assert(txVerification.txId === fetchedEvent.txId, 'Transaction ID mismatch');
    
    console.log('✅ Transaction verified');
    console.log(`   Block Number: ${txVerification.blockNumber}`);
    console.log(`   Timestamp: ${txVerification.timestamp}`);
    
    // Test 5: Generate verification certificate
    console.log('\nTest 5: Generating verification certificate');
    const certificate = await blockchain.generateVerificationCertificate(eventResult.eventId);
    
    assert(certificate, 'Failed to generate certificate');
    assert(certificate.eventId === eventResult.eventId, 'Certificate event ID mismatch');
    assert(certificate.transactionId === fetchedEvent.txId, 'Certificate transaction ID mismatch');
    assert(certificate.status === 'VERIFIED', 'Certificate not verified');
    
    console.log('✅ Verification certificate generated successfully');
    
    // Test 6: Tamper detection
    console.log('\nTest 6: Testing tamper detection');
    
    // Create a tampered event by copying the original and modifying a field
    const tamperedEvent = JSON.parse(JSON.stringify(fetchedEvent));
    tamperedEvent.details = JSON.stringify({
      ...JSON.parse(tamperedEvent.details),
      tamperedField: 'This was modified'
    });
    
    const tamperedIntegrityResult = blockchain.verifyEventIntegrity(tamperedEvent);
    
    assert(!tamperedIntegrityResult.isValid, 'Tampered event incorrectly passed integrity check');
    
    console.log('✅ Tamper detection works correctly');
    
    console.log('\nAll blockchain verification tests passed! ✅');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

// Run the tests
runTests().catch(err => {
  console.error('Error running tests:', err);
  process.exit(1);
}); 