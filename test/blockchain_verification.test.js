/**
 * NeuraShield Blockchain Verification Tests
 */

// Import mocks
jest.mock('../backend/src/blockchain-adapter', () => require('./mocks/blockchain-adapter.mock'));

const { getBlockchainImplementation } = require('../backend/src/blockchain-adapter');

describe('Blockchain Verification', () => {
  let blockchain;
  let testEvent;

  beforeAll(async () => {
    blockchain = await getBlockchainImplementation();
    await blockchain.initBlockchain();
  });

  beforeEach(() => {
    testEvent = {
      id: `test-event-${Date.now()}`,
      timestamp: new Date().toISOString(),
      type: 'TEST_EVENT',
      details: {
        test_field: 'Test value',
        severity: 'medium',
        source: 'verification_test'
      }
    };
  });

  test('should process a security event', async () => {
    // Test 1: Process security event
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    
    expect(eventResult.success).toBe(true);
    expect(eventResult.eventId).toBe(testEvent.id);
    expect(eventResult.txId).toBeDefined();
    expect(eventResult.eventHash).toBeDefined();
  });

  test('should fetch an event by ID', async () => {
    // Test 2: Fetch event
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    const fetchedEvent = await blockchain.fetchEvent(eventResult.eventId);
    
    expect(fetchedEvent).toBeDefined();
    expect(fetchedEvent.id).toBe(eventResult.eventId);
    expect(fetchedEvent.eventHash).toBe(eventResult.eventHash);
  });

  test('should verify event integrity', async () => {
    // Test 3: Verify event integrity
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    const fetchedEvent = await blockchain.fetchEvent(eventResult.eventId);
    const integrityResult = blockchain.verifyEventIntegrity(fetchedEvent);
    
    expect(integrityResult.isValid).toBe(true);
    expect(integrityResult.calculatedHash).toBe(fetchedEvent.eventHash);
  });

  test('should verify a transaction', async () => {
    // Test 4: Verify transaction
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    const fetchedEvent = await blockchain.fetchEvent(eventResult.eventId);
    const txVerification = await blockchain.verifyTransaction(fetchedEvent.txId);
    
    expect(txVerification.isValid).toBe(true);
    expect(txVerification.txId).toBe(fetchedEvent.txId);
    expect(txVerification.blockNumber).toBeDefined();
    expect(txVerification.timestamp).toBeDefined();
  });

  test('should generate a verification certificate', async () => {
    // Test 5: Generate verification certificate
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    const certificate = await blockchain.generateVerificationCertificate(eventResult.eventId);
    
    expect(certificate).toBeDefined();
    expect(certificate.eventId).toBe(eventResult.eventId);
    expect(certificate.transactionId).toBe(eventResult.txId);
    expect(certificate.status).toBe('VERIFIED');
  });

  test('should detect tampered events', async () => {
    // Test 6: Tamper detection
    const eventResult = await blockchain.processSecurityEvent(testEvent);
    const fetchedEvent = await blockchain.fetchEvent(eventResult.eventId);
    
    // Create a tampered event by copying the original and modifying a field
    const tamperedEvent = JSON.parse(JSON.stringify(fetchedEvent));
    tamperedEvent.details = JSON.stringify({
      ...JSON.parse(tamperedEvent.details),
      tamperedField: 'This was modified'
    });
    
    const tamperedIntegrityResult = blockchain.verifyEventIntegrity(tamperedEvent);
    
    expect(tamperedIntegrityResult.isValid).toBe(false);
  });
}); 