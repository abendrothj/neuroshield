/**
 * NeuraShield Blockchain Adapter
 * 
 * This adapter handles connections to the blockchain systems,
 * with logic to decide between actual Hyperledger Fabric and 
 * a fallback implementation for development/testing.
 */

const fs = require('fs');
const path = require('path');
const winston = require('winston');
const { Gateway, Wallets } = require('fabric-network');
const crypto = require('crypto');

// Setup logging
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ level, message, timestamp }) => {
            return `${timestamp} ${level}: ${message}`;
        })
    ),
    defaultMeta: { service: 'blockchain-adapter' },
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ 
            filename: path.join(__dirname, '..', 'logs/blockchain-adapter.log') 
        })
    ]
});

// Configuration
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'neurashield-channel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'neurashield';

// Get absolute path to connection profile
const ccpPath = path.resolve(__dirname, '..', 'connection-profile.json');

/**
 * Configure the blockchain to use ONLY real Fabric network
 * and never fall back to mock implementation.
 */
async function getBlockchainImplementation() {
    logger.info('Getting blockchain implementation (PRODUCTION ONLY MODE)');
    
    try {
        // Check if connection profile exists
        if (!fs.existsSync(ccpPath)) {
            throw new Error('Connection profile not found at: ' + ccpPath);
        }
        
        // Read the connection profile
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));
        logger.info('Connection profile loaded successfully');
        
        // Return the real blockchain implementation
        return {
            /**
             * Initialize the blockchain connection.
             */
            async initBlockchain() {
                logger.info('Initializing real blockchain connection');
                try {
                    // Check wallet exists, create if not
                    const walletPath = path.join(__dirname, '..', 'wallet');
                    if (!fs.existsSync(walletPath)) {
                        fs.mkdirSync(walletPath, { recursive: true });
                    }
                    
                    // Get wallet and check admin identity exists
                    const wallet = await Wallets.newFileSystemWallet(walletPath);
                    const adminExists = await wallet.get('admin');
                    
                    if (!adminExists) {
                        logger.warn('Admin identity not found in wallet, production systems should have identity management');
                        
                        // In production, you would not create the identity here
                        // This is a placeholder for testing only
                        logger.info('Creating a test admin identity (FOR TESTING ONLY)');
                        const adminIdentity = {
                            credentials: {
                                certificate: fs.readFileSync(path.resolve(__dirname, '..', '..', 'fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/signcerts/cert.pem')).toString(),
                                privateKey: fs.readFileSync(path.resolve(__dirname, '..', '..', 'fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore/priv_sk')).toString(),
                            },
                            type: 'X.509',
                        };
                        await wallet.put('admin', adminIdentity);
                    }
                    
                    logger.info('Real blockchain initialized successfully');
                    return true;
                } catch (error) {
                    logger.error(`Error initializing blockchain: ${error.message}`);
                    throw error;
                }
            },
            
            /**
             * Process a security event and record it on the blockchain.
             */
            async processSecurityEvent(eventData) {
                logger.info(`Processing security event with real blockchain: ${eventData.id || 'new-event'}`);
                try {
                    const walletPath = path.join(__dirname, '..', 'wallet');
                    const wallet = await Wallets.newFileSystemWallet(walletPath);
                    
                    // Get Gateway connection
                    const gateway = new Gateway();
                    await gateway.connect(ccp, { 
                        wallet, 
                        identity: 'admin', 
                        discovery: { enabled: false }
                    });
                    
                    // Get network and contract
                    const network = await gateway.getNetwork(CHANNEL_NAME);
                    const contract = network.getContract(CHAINCODE_NAME);
                    
                    // Create event ID if not provided
                    const eventId = eventData.id || `event-${Date.now()}`;
                    const timestamp = eventData.timestamp || new Date().toISOString();
                    const type = eventData.type || 'SecurityAlert';
                    const details = typeof eventData.details === 'string' 
                        ? eventData.details 
                        : JSON.stringify(eventData.details || {});
                    
                    // Generate a hash of the event data for verification
                    const eventHash = this.generateEventHash(eventId, timestamp, type, details);
                    
                    // Submit transaction to record event
                    const result = await contract.submitTransaction(
                        'LogEvent', 
                        eventId, 
                        timestamp, 
                        type, 
                        details,
                        eventData.ipfsHash || '',
                        eventHash
                    );
                    
                    // Parse transaction result to get transaction ID
                    const txId = JSON.parse(result.toString()).txId || '';
                    
                    // Close gateway connection
                    gateway.disconnect();
                    
                    logger.info(`Event ${eventId} successfully recorded on blockchain with txId ${txId}`);
                    return { 
                        success: true, 
                        eventId,
                        txId,
                        eventHash
                    };
                } catch (error) {
                    logger.error(`Error recording event on blockchain: ${error.message}`);
                    throw error;
                }
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
                        event.eventId,
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
                    logger.error(`Error verifying event integrity: ${error.message}`);
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
                logger.info('Fetching all events from real blockchain');
                try {
                    const walletPath = path.join(__dirname, '..', 'wallet');
                    const wallet = await Wallets.newFileSystemWallet(walletPath);
                    
                    // Get Gateway connection
                    const gateway = new Gateway();
                    await gateway.connect(ccp, { 
                        wallet, 
                        identity: 'admin', 
                        discovery: { enabled: false }
                    });
                    
                    // Get network and contract
                    const network = await gateway.getNetwork(CHANNEL_NAME);
                    const contract = network.getContract(CHAINCODE_NAME);
                    
                    // Query all events
                    const result = await contract.evaluateTransaction('QueryAllEvents');
                    const events = JSON.parse(result.toString());
                    
                    // Close gateway connection
                    gateway.disconnect();
                    
                    logger.info(`Retrieved ${events.length} events from blockchain`);
                    return { events };
                } catch (error) {
                    logger.error(`Error fetching events from blockchain: ${error.message}`);
                    throw error;
                }
            },
            
            /**
             * Fetch a specific event from the blockchain by ID.
             */
            async fetchEvent(eventId) {
                logger.info(`Fetching event ${eventId} from real blockchain`);
                try {
                    const walletPath = path.join(__dirname, '..', 'wallet');
                    const wallet = await Wallets.newFileSystemWallet(walletPath);
                    
                    // Get Gateway connection
                    const gateway = new Gateway();
                    await gateway.connect(ccp, { 
                        wallet, 
                        identity: 'admin', 
                        discovery: { enabled: false }
                    });
                    
                    // Get network and contract
                    const network = await gateway.getNetwork(CHANNEL_NAME);
                    const contract = network.getContract(CHAINCODE_NAME);
                    
                    // Query specific event
                    const result = await contract.evaluateTransaction('QueryEvent', eventId);
                    const event = JSON.parse(result.toString());
                    
                    // Close gateway connection
                    gateway.disconnect();
                    
                    logger.info(`Retrieved event ${eventId} from blockchain`);
                    return event;
                } catch (error) {
                    logger.error(`Error fetching event ${eventId} from blockchain: ${error.message}`);
                    throw error;
                }
            },
            
            /**
             * Verify a blockchain transaction by ID.
             */
            async verifyTransaction(txId) {
                logger.info(`Verifying transaction ${txId}`);
                try {
                    const walletPath = path.join(__dirname, '..', 'wallet');
                    const wallet = await Wallets.newFileSystemWallet(walletPath);
                    
                    // Get Gateway connection
                    const gateway = new Gateway();
                    await gateway.connect(ccp, { 
                        wallet, 
                        identity: 'admin', 
                        discovery: { enabled: false }
                    });
                    
                    // Get network
                    const network = await gateway.getNetwork(CHANNEL_NAME);
                    
                    // Get transaction details
                    const transaction = await network.getChannel().queryTransaction(txId);
                    
                    // Close gateway connection
                    gateway.disconnect();
                    
                    if (!transaction) {
                        throw new Error(`Transaction ${txId} not found`);
                    }
                    
                    // Extract validation code
                    const validationCode = transaction.transactionEnvelope.validationCode;
                    const isValid = validationCode === 0; // 0 means valid in Fabric
                    
                    // Get timestamp
                    const timestamp = new Date(transaction.transactionEnvelope.header.channelHeader.timestamp).toISOString();
                    
                    // Get block information
                    const blockNumber = transaction.transactionEnvelope.blockNumber || 'unknown';
                    
                    logger.info(`Transaction ${txId} verification: ${isValid ? 'Valid' : 'Invalid'}`);
                    
                    return {
                        isValid,
                        txId,
                        blockNumber,
                        timestamp,
                        validationCode
                    };
                } catch (error) {
                    logger.error(`Error verifying transaction ${txId}: ${error.message}`);
                    throw error;
                }
            },
            
            /**
             * Process AI detection and record it on the blockchain.
             */
            async aiDetectionWebhook(req, res) {
                logger.info('Processing AI detection webhook with real blockchain');
                try {
                    const prediction = req.body.prediction;
                    if (!prediction) {
                        return res.status(400).json({ error: 'Missing prediction data' });
                    }
                    
                    // Create event from prediction
                    const eventData = {
                        id: `ai-detection-${Date.now()}`,
                        timestamp: prediction.timestamp || new Date().toISOString(),
                        type: 'AI_DETECTION',
                        details: {
                            threat_type: prediction.threat_type,
                            confidence: prediction.confidence,
                            affected_system: prediction.affected_system,
                            source_ip: prediction.source_ip
                        }
                    };
                    
                    // Process the event
                    const result = await this.processSecurityEvent(eventData);
                    
                    res.json({
                        success: true,
                        event_id: result.eventId,
                        transaction_id: result.txId,
                        hash: result.eventHash,
                        blockchain: 'hyperledger-fabric'
                    });
                } catch (error) {
                    logger.error(`Error in AI detection webhook: ${error.message}`);
                    res.status(500).json({ error: error.message });
                }
            },
            
            /**
             * Generate a verification certificate for an event.
             */
            async generateVerificationCertificate(eventId) {
                logger.info(`Generating verification certificate for event ${eventId}`);
                try {
                    // Fetch the event
                    const event = await this.fetchEvent(eventId);
                    if (!event) {
                        throw new Error(`Event ${eventId} not found`);
                    }
                    
                    // Verify event integrity
                    const integrityCheck = this.verifyEventIntegrity(event);
                    if (!integrityCheck.isValid) {
                        throw new Error('Event data integrity check failed');
                    }
                    
                    // Verify transaction
                    const txVerification = await this.verifyTransaction(event.txId);
                    if (!txVerification.isValid) {
                        throw new Error('Transaction verification failed');
                    }
                    
                    // Generate certificate
                    const certificate = {
                        eventId: event.eventId,
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
                    
                    logger.info(`Verification certificate generated for event ${eventId}`);
                    return certificate;
                    
                } catch (error) {
                    logger.error(`Error generating verification certificate: ${error.message}`);
                    throw error;
                }
            }
        };
    } catch (error) {
        logger.error(`Error getting blockchain implementation: ${error.message}`);
        throw error;
    }
}

module.exports = {
    getBlockchainImplementation
}; 