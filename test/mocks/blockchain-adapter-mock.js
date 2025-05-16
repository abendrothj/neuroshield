/**
 * NeuraShield Blockchain Adapter Mock
 * 
 * This is a mock implementation of the blockchain adapter for testing purposes.
 * It simulates the behavior of a real blockchain without requiring an actual network.
 */

const crypto = require('crypto');

// In-memory storage for mock blockchain data
const eventStore = new Map();
const transactionStore = new Map();

// Mock blockchain implementation
const getMockBlockchainImplementation = async () => {
  return {
    /**
     * Initialize the blockchain connection.
     */
    async initBlockchain() {
      console.log('Initializing mock blockchain connection');
      return true;
    },
    
    /**
     * Process a security event and record it on the blockchain.
     */
    async processSecurityEvent(eventData) {
      console.log(`Processing security event in mock blockchain: ${eventData.id || 'new-event'}`);
      
      // Create event ID if not provided
      const eventId = eventData.id || `event-${Date.now()}`;
      const timestamp = eventData.timestamp || new Date().toISOString();
      const type = eventData.type || 'SecurityAlert';
      const details = typeof eventData.details === 'string' 
          ? eventData.details 
          : JSON.stringify(eventData.details || {});
      
      // Generate a hash of the event data
      const eventHash = this.generateEventHash(eventId, timestamp, type, details);
      
      // Generate mock transaction ID
      const txId = `tx-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      
      // Store the event
      const event = {
        id: eventId,
        timestamp,
        type,
        details,
        ipfsHash: eventData.ipfsHash || '',
        eventHash,
        txId,
        blockNumber: Math.floor(Date.now() / 1000)
      };
      
      eventStore.set(eventId, event);
      
      // Store the transaction
      const transaction = {
        txId,
        validationCode: 0,
        timestamp: new Date().toISOString(),
        blockNumber: event.blockNumber
      };
      
      transactionStore.set(txId, transaction);
      
      console.log(`Event ${eventId} successfully recorded on mock blockchain with txId ${txId}`);
      return { 
        success: true, 
        eventId,
        txId,
        eventHash
      };
    },
    
    /**
     * Generate a cryptographic hash of event data for verification.
     */
    generateEventHash(eventId, timestamp, type, details) {
      const data = `${eventId}|${timestamp}|${type}|${details}`;
      return crypto.createHash('sha256').update(data).digest('hex');
    },
    
    /**
     * Verify if an event's data matches its recorded hash on the blockchain.
     */
    verifyEventIntegrity(event) {
      try {
        // Re-generate the hash from the event data
        const calculatedHash = this.generateEventHash(
          event.eventId || event.id,
          event.timestamp,
          event.type,
          event.details
        );
        
        // Compare with the stored hash
        const isValid = calculatedHash === event.eventHash;
        
        return {
          isValid,
          calculatedHash,
          storedHash: event.eventHash
        };
      } catch (error) {
        console.error(`Error verifying event integrity: ${error.message}`);
        return {
          isValid: false,
          error: error.message
        };
      }
    },
    
    /**
     * Fetch all events from the blockchain.
     */
    async fetchEvents() {
      console.log('Fetching all events from mock blockchain');
      
      const events = Array.from(eventStore.values());
      
      console.log(`Retrieved ${events.length} events from mock blockchain`);
      return { events };
    },
    
    /**
     * Fetch a specific event from the blockchain by ID.
     */
    async fetchEvent(eventId) {
      console.log(`Fetching event ${eventId} from mock blockchain`);
      
      const event = eventStore.get(eventId);
      
      if (!event) {
        throw new Error(`Event ${eventId} not found`);
      }
      
      console.log(`Retrieved event ${eventId} from mock blockchain`);
      return event;
    },
    
    /**
     * Verify a blockchain transaction by ID.
     */
    async verifyTransaction(txId) {
      console.log(`Verifying transaction ${txId} in mock blockchain`);
      
      const transaction = transactionStore.get(txId);
      
      if (!transaction) {
        throw new Error(`Transaction ${txId} not found`);
      }
      
      // Extract validation code (0 means valid)
      const validationCode = transaction.validationCode;
      const isValid = validationCode === 0;
      
      console.log(`Transaction ${txId} verification: ${isValid ? 'Valid' : 'Invalid'}`);
      
      return {
        isValid,
        txId,
        blockNumber: transaction.blockNumber,
        timestamp: transaction.timestamp,
        validationCode
      };
    },
    
    /**
     * Generate a verification certificate for an event.
     */
    async generateVerificationCertificate(eventId) {
      console.log(`Generating verification certificate for event ${eventId} in mock blockchain`);
      
      // Fetch the event
      const event = await this.fetchEvent(eventId);
      
      // Verify event integrity
      const integrityCheck = this.verifyEventIntegrity(event);
      
      // Verify transaction
      const txVerification = await this.verifyTransaction(event.txId);
      
      // Generate certificate
      const certificate = {
        eventId: event.id,
        type: event.type,
        timestamp: event.timestamp,
        blockchainTimestamp: txVerification.timestamp,
        blockNumber: txVerification.blockNumber,
        transactionId: event.txId,
        dataHash: event.eventHash,
        verificationTimestamp: new Date().toISOString(),
        status: 'VERIFIED',
        verificationDetails: {
          dataIntegrity: integrityCheck,
          transactionValidity: txVerification
        }
      };
      
      console.log(`Verification certificate generated for event ${eventId}`);
      return certificate;
    }
  };
};

module.exports = {
  getBlockchainImplementation: getMockBlockchainImplementation
}; 