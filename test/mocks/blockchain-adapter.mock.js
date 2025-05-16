/**
 * Mock Blockchain Adapter for testing
 */

const crypto = require('crypto');

class MockBlockchainAdapter {
  constructor() {
    // In-memory storage for testing
    this.events = new Map();
    this.transactions = new Map();
    this.network = 'test-network';
    this.initialized = false;
  }

  async initBlockchain() {
    this.initialized = true;
    return { success: true };
  }

  async processSecurityEvent(event) {
    // Generate a hash for the event
    const eventHash = this._generateHash(JSON.stringify(event));
    
    // Create a transaction ID
    const txId = `tx_${crypto.randomBytes(8).toString('hex')}`;
    
    // Store the event
    const storedEvent = {
      id: event.id,
      timestamp: event.timestamp,
      type: event.type,
      details: JSON.stringify(event.details),
      eventHash,
      txId,
      blockNumber: Math.floor(Math.random() * 1000),
      createdAt: new Date().toISOString()
    };
    
    this.events.set(event.id, storedEvent);
    
    // Store transaction info
    this.transactions.set(txId, {
      txId,
      blockNumber: storedEvent.blockNumber,
      timestamp: new Date().toISOString(),
      eventId: event.id,
    });
    
    return {
      success: true,
      eventId: event.id,
      txId,
      eventHash,
    };
  }

  async fetchEvent(eventId) {
    const event = this.events.get(eventId);
    if (!event) {
      throw new Error(`Event with ID ${eventId} not found`);
    }
    return event;
  }

  verifyEventIntegrity(event) {
    // Calculate hash of the event
    const detailsObj = typeof event.details === 'string' 
      ? JSON.parse(event.details) 
      : event.details;
      
    const eventToHash = {
      id: event.id,
      timestamp: event.timestamp,
      type: event.type,
      details: detailsObj
    };
    
    const calculatedHash = this._generateHash(JSON.stringify(eventToHash));
    const isValid = calculatedHash === event.eventHash;
    
    return {
      isValid,
      calculatedHash,
      storedHash: event.eventHash
    };
  }

  async verifyTransaction(txId) {
    const tx = this.transactions.get(txId);
    if (!tx) {
      throw new Error(`Transaction with ID ${txId} not found`);
    }
    
    return {
      isValid: true,
      txId: tx.txId,
      blockNumber: tx.blockNumber,
      timestamp: tx.timestamp,
      network: this.network
    };
  }

  async generateVerificationCertificate(eventId) {
    const event = await this.fetchEvent(eventId);
    const tx = this.transactions.get(event.txId);
    
    return {
      certificateId: `cert_${crypto.randomBytes(8).toString('hex')}`,
      eventId: event.id,
      transactionId: event.txId,
      blockNumber: tx.blockNumber,
      network: this.network,
      timestamp: new Date().toISOString(),
      status: 'VERIFIED',
      issuer: 'NeuraShield Test Network',
      integrityCheck: 'PASSED'
    };
  }

  _generateHash(data) {
    return crypto.createHash('sha256').update(data).digest('hex');
  }
}

// Export factory function
function getBlockchainImplementation() {
  return Promise.resolve(new MockBlockchainAdapter());
}

module.exports = { getBlockchainImplementation, MockBlockchainAdapter }; 