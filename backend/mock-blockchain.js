/**
 * NeuraShield Mock Blockchain Integration
 * 
 * This module provides mock implementations of the blockchain integration functions
 * for testing purposes when a real Hyperledger Fabric network is not available.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const winston = require('winston');

// Configure logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'mock-blockchain' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ 
      filename: path.join(__dirname, 'logs/mock-blockchain.log')
    })
  ]
});

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// In-memory storage for events
const mockLedger = {
  events: {},
  getAllEvents: function() {
    return Object.values(this.events);
  },
  getEvent: function(id) {
    return this.events[id] || null;
  },
  addEvent: function(event) {
    this.events[event.ID] = event;
    return true;
  }
};

// Mock storage for IPFS data
const mockIPFS = {
  files: {},
  add: async function(content) {
    const hash = 'Qm' + crypto.createHash('sha256').update(content).digest('hex').substring(0, 44);
    this.files[hash] = content;
    return hash;
  },
  get: async function(hash) {
    return this.files[hash] || null;
  }
};

// Mock identity operations
const mockIdentity = {
  users: {
    admin: {
      certificate: 'MOCK_ADMIN_CERT',
      privateKey: 'MOCK_ADMIN_KEY',
      mspId: 'Org1MSP'
    }
  },
  exists: function(userId) {
    return this.users[userId] !== undefined;
  },
  register: function(userId, role, affiliation) {
    if (this.exists(userId)) {
      return Promise.resolve(false);
    }
    this.users[userId] = {
      certificate: `MOCK_CERT_${userId}`,
      privateKey: `MOCK_KEY_${userId}`,
      mspId: 'Org1MSP',
      role: role,
      affiliation: affiliation
    };
    return Promise.resolve(true);
  },
  delete: function(userId) {
    if (!this.exists(userId)) {
      return Promise.resolve(false);
    }
    delete this.users[userId];
    return Promise.resolve(true);
  },
  list: function() {
    return Object.keys(this.users);
  }
};

/**
 * Initialize the mock blockchain environment
 */
async function initBlockchain() {
  logger.info('Initializing mock blockchain environment');
  
  // Add some initial events to the ledger for testing
  const initEvent = {
    ID: 'init1',
    Timestamp: new Date().toISOString(),
    Type: 'Initialization',
    Details: JSON.stringify({ message: 'Mock blockchain initialized' }),
    IPFSHash: ''
  };
  
  mockLedger.addEvent(initEvent);
  logger.info('Mock blockchain initialized with genesis event');
  
  return true;
}

/**
 * Process a security event and add it to the mock ledger
 * @param {Object} threatData - The threat data to process
 * @returns {Promise<Object>} - The result with eventId and ipfsHash
 */
async function processSecurityEvent(threatData) {
  try {
    // Generate unique event ID
    const eventId = `event-${crypto.randomBytes(4).toString('hex')}-${Date.now()}`;
    
    // Format timestamp
    const timestamp = new Date().toISOString();
    
    // Format event type based on threat level
    let eventType = 'Info';
    if (threatData.confidence > 0.9) {
      eventType = 'Critical';
    } else if (threatData.confidence > 0.7) {
      eventType = 'High';
    } else if (threatData.confidence > 0.4) {
      eventType = 'Medium';
    } else if (threatData.confidence > 0.2) {
      eventType = 'Low';
    }
    
    // Prepare full event details
    const fullEventDetails = {
      ...threatData,
      processed_timestamp: timestamp,
      detection_source: 'NeuraShield AI',
      system_version: process.env.SYSTEM_VERSION || '1.0.0'
    };
    
    // Store in mock IPFS if enabled
    let ipfsHash = '';
    if (process.env.IPFS_ENABLED === 'true') {
      ipfsHash = await mockIPFS.add(JSON.stringify(fullEventDetails));
      logger.info(`Stored data in mock IPFS with hash: ${ipfsHash}`);
    }
    
    // Prepare event summary
    const eventSummary = JSON.stringify({
      threat_type: threatData.threat_type,
      confidence: threatData.confidence,
      severity: eventType,
      detection_time: timestamp
    });
    
    // Create the event object
    const event = {
      ID: eventId,
      Timestamp: timestamp,
      Type: eventType,
      Details: eventSummary,
      IPFSHash: ipfsHash
    };
    
    // Add to mock ledger
    mockLedger.addEvent(event);
    logger.info(`Added event ${eventId} to mock ledger`);
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 100));
    
    return { success: true, eventId, ipfsHash };
  } catch (error) {
    logger.error(`Error processing security event: ${error.message}`);
    throw error;
  }
}

/**
 * AI detection webhook handler
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function aiDetectionWebhook(req, res) {
  try {
    const threatData = req.body;
    
    // Validate the incoming data
    if (!threatData || !threatData.threat_type || !threatData.confidence) {
      logger.error('Invalid threat data received');
      return res.status(400).json({ error: 'Invalid threat data' });
    }
    
    // Process the event asynchronously
    processSecurityEvent(threatData)
      .then(result => {
        logger.info(`Successfully processed event: ${result.eventId}`);
      })
      .catch(error => {
        logger.error(`Error in webhook processing: ${error.message}`);
      });
    
    // Respond immediately to avoid blocking the AI system
    res.status(202).json({ message: 'Event received for processing' });
    
  } catch (error) {
    logger.error(`Error in AI detection webhook: ${error.message}`);
    res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Fetch all events from the mock ledger
 */
async function fetchEvents() {
  return mockLedger.getAllEvents();
}

/**
 * Fetch a specific event by ID
 * @param {string} eventId - The ID of the event to fetch
 */
async function fetchEvent(eventId) {
  return mockLedger.getEvent(eventId);
}

/**
 * Mock user identity verification
 * @param {string} identityLabel - The identity to verify
 */
async function verifyIdentity(identityLabel) {
  return mockIdentity.exists(identityLabel);
}

/**
 * Mock user identity registration
 * @param {string} userId - The user ID to register
 * @param {string} userRole - The role for the user
 * @param {string} affiliation - The user's affiliation
 */
async function registerUser(userId, userRole, affiliation) {
  return await mockIdentity.register(userId, userRole, affiliation);
}

/**
 * List all mock identities
 */
async function listIdentities() {
  return mockIdentity.list();
}

/**
 * Delete a mock identity
 * @param {string} identityLabel - The identity to delete
 */
async function deleteIdentity(identityLabel) {
  return await mockIdentity.delete(identityLabel);
}

/**
 * Initialize identities
 */
async function initializeIdentities() {
  logger.info('Initializing mock identities');
  return true;
}

/**
 * Poll for AI detections
 */
async function fetchAIDetections() {
  logger.info('Mock polling for AI detections');
  return [];
}

/**
 * Initialize polling
 */
function initPolling() {
  logger.info('Mock polling initialized');
}

module.exports = {
  initBlockchain,
  processSecurityEvent,
  aiDetectionWebhook,
  fetchEvents,
  fetchEvent,
  verifyIdentity,
  registerUser,
  listIdentities,
  deleteIdentity,
  initializeIdentities,
  fetchAIDetections,
  initPolling
}; 