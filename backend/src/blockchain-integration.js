/**
 * NeuraShield Blockchain Integration Service
 * 
 * This module provides the integration layer between the AI detection system and the 
 * Hyperledger Fabric blockchain for immutable logging of security events.
 */

const { Gateway, Wallets } = require('fabric-network');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const Queue = require('better-queue');
const crypto = require('crypto');
const ipfsClient = require('ipfs-http-client');
const winston = require('winston');
const identityManager = require('../identity-manager');

// Configure logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'blockchain-integration' },
  transports: [
    new winston.transports.File({ filename: path.join(__dirname, '../logs/blockchain-error.log'), level: 'error' }),
    new winston.transports.File({ filename: path.join(__dirname, '../logs/blockchain-combined.log') }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Configuration from environment variables
const AI_API_URL = process.env.AI_API_URL || 'http://localhost:8000';
const CONNECTION_PROFILE_PATH = process.env.CONNECTION_PROFILE_PATH || path.resolve(__dirname, '../connection-profile.json');
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'neurashield-channel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'neurashield';
const WALLET_PATH = process.env.WALLET_PATH || path.join(__dirname, '../wallet');
const IDENTITY = process.env.BLOCKCHAIN_IDENTITY || 'admin';
const IPFS_API = process.env.IPFS_API || 'http://localhost:5001/api/v0';
const IPFS_GATEWAY = process.env.IPFS_GATEWAY || 'http://localhost:8080/ipfs/';
const EVENT_QUEUE_CONCURRENCY = parseInt(process.env.EVENT_QUEUE_CONCURRENCY || '5');
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '5');
const RETRY_DELAY = parseInt(process.env.RETRY_DELAY || '5000');
const MSP_ID = process.env.MSP_ID || 'Org1MSP';

// Initialize IPFS client if enabled
let ipfs = null;
if (process.env.IPFS_ENABLED === 'true') {
  try {
    ipfs = ipfsClient(IPFS_API);
    logger.info('IPFS client initialized');
  } catch (error) {
    logger.error(`Failed to initialize IPFS client: ${error.message}`);
  }
}

/**
 * Process queue for handling blockchain submissions with retry logic
 */
const eventQueue = new Queue(async (event, callback) => {
  try {
    const result = await submitToBlockchain(event);
    callback(null, result);
  } catch (error) {
    logger.error(`Error submitting to blockchain: ${error.message}`);
    callback(error);
  }
}, { 
  concurrent: EVENT_QUEUE_CONCURRENCY,
  maxRetries: MAX_RETRIES,
  retryDelay: RETRY_DELAY
});

/**
 * Submit a security event to the blockchain
 * @param {Object} event - The security event to be logged
 * @returns {Promise} - Promise with the transaction result
 */
async function submitToBlockchain(event) {
  let gateway;

  try {
    // Get gateway connection using admin identity directly
    gateway = await identityManager.getGatewayConnection('admin', {
      discovery: { enabled: false },
      clientTlsIdentity: 'admin',
      mspId: MSP_ID
    });
    
    // Get the network and contract
    const network = await gateway.getNetwork(CHANNEL_NAME);
    const contract = network.getContract(CHAINCODE_NAME);
    
    // Submit the transaction to log the event
    logger.info(`Submitting event to blockchain: ${event.id}`);
    const result = await contract.submitTransaction(
      'LogEvent',
      event.id,
      event.timestamp,
      event.type,
      event.details,
      event.ipfsHash || ''
    );
    
    logger.info(`Event ${event.id} successfully logged to blockchain`);
    return result;
    
  } catch (error) {
    logger.error(`Error in blockchain submission: ${error.message}`);
    if (error.message.includes('MVCC_READ_CONFLICT')) {
      logger.warn('MVCC read conflict detected, will retry transaction');
    }
    throw error;
  } finally {
    // Disconnect from the gateway
    if (gateway) {
      gateway.disconnect();
    }
  }
}

/**
 * Store detailed event data in IPFS
 * @param {Object} eventData - Full event data to store in IPFS
 * @returns {Promise<String>} - IPFS content identifier (CID)
 */
async function storeInIPFS(eventData) {
  try {
    if (!ipfs) {
      logger.warn('IPFS client not initialized, skipping IPFS storage');
      return '';
    }
    
    const content = JSON.stringify(eventData);
    const { cid } = await ipfs.add(content);
    logger.info(`Event data stored in IPFS with CID: ${cid.toString()}`);
    return cid.toString();
  } catch (error) {
    logger.error(`Error storing in IPFS: ${error.message}`);
    return '';
  }
}

/**
 * Process a security event from the AI detection system
 * @param {Object} threatData - Threat data from the AI system
 * @returns {Promise<Object>} - Result of the blockchain submission
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
    
    // Prepare full event details including AI metadata
    const fullEventDetails = {
      ...threatData,
      processed_timestamp: timestamp,
      detection_source: 'NeuraShield AI',
      system_version: process.env.SYSTEM_VERSION || '1.0.0'
    };
    
    // Store full details in IPFS if enabled
    let ipfsHash = '';
    if (process.env.IPFS_ENABLED === 'true') {
      ipfsHash = await storeInIPFS(fullEventDetails);
    }
    
    // Prepare event summary for blockchain
    const eventSummary = JSON.stringify({
      threat_type: threatData.threat_type,
      confidence: threatData.confidence,
      severity: eventType,
      detection_time: timestamp
    });
    
    // Prepare the event object
    const event = {
      id: eventId,
      timestamp: timestamp,
      type: eventType,
      details: eventSummary,
      ipfsHash: ipfsHash
    };
    
    // Add to processing queue with retry capability
    return new Promise((resolve, reject) => {
      eventQueue.push(event, (err, result) => {
        if (err) {
          logger.error(`Failed to process event after retries: ${err.message}`);
          reject(err);
        } else {
          resolve({ success: true, eventId, ipfsHash });
        }
      });
    });
    
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
 * Periodic fetching of AI detections (polling method)
 */
async function fetchAIDetections() {
  try {
    const response = await axios.get(`${AI_API_URL}/api/recent-threats`);
    
    if (response.data && Array.isArray(response.data.threats)) {
      const threats = response.data.threats;
      logger.info(`Fetched ${threats.length} threats from AI API`);
      
      // Process each threat
      for (const threat of threats) {
        await processSecurityEvent(threat)
          .then(result => {
            logger.info(`Successfully processed fetched event: ${result.eventId}`);
          })
          .catch(error => {
            logger.error(`Error processing fetched event: ${error.message}`);
          });
      }
    }
  } catch (error) {
    logger.error(`Error fetching AI detections: ${error.message}`);
  }
}

/**
 * Initialize periodic polling if webhook is not enabled
 */
function initPolling() {
  const pollingInterval = parseInt(process.env.POLLING_INTERVAL_MS || '60000');
  logger.info(`Initializing polling for AI detections every ${pollingInterval}ms`);
  setInterval(fetchAIDetections, pollingInterval);
}

/**
 * Initialize the blockchain connection and verify it's working
 */
async function initBlockchain() {
  try {
    // Initialize identities using the identity manager
    const identitiesInitialized = await identityManager.initializeIdentities();
    if (!identitiesInitialized) {
      logger.error('Failed to initialize identities');
      return false;
    }
    
    // Verify the blockchain service identity
    const serviceId = process.env.BLOCKCHAIN_IDENTITY || 'blockchain-service';
    const serviceIdentityValid = await identityManager.verifyIdentity(serviceId);
    
    if (!serviceIdentityValid) {
      logger.error(`Service identity ${serviceId} is not valid`);
      
      // Try admin as fallback
      logger.info('Trying admin identity as fallback');
      const adminValid = await identityManager.verifyIdentity('admin');
      
      if (!adminValid) {
        logger.error('Admin identity is also not valid');
        return false;
      }
    }
    
    // Test a blockchain query to verify connectivity
    const gateway = await identityManager.getGatewayConnection(
      serviceIdentityValid ? serviceId : 'admin'
    );
    
    const network = await gateway.getNetwork(CHANNEL_NAME);
    const contract = network.getContract(CHAINCODE_NAME);
    
    // Try to get all events as a test
    await contract.evaluateTransaction('QueryAllEvents');
    
    logger.info('Successfully connected to the blockchain network');
    gateway.disconnect();
    
    return true;
  } catch (error) {
    logger.error(`Failed to initialize blockchain connection: ${error.message}`);
    return false;
  }
}

module.exports = {
  initBlockchain,
  initPolling,
  processSecurityEvent,
  aiDetectionWebhook,
  fetchAIDetections
}; 