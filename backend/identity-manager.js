/**
 * NeuraShield Identity Management System
 * 
 * This module handles identity management for the Hyperledger Fabric blockchain,
 * including enrollment, registration, and verification of user identities.
 */

const fs = require('fs');
const path = require('path');
const FabricCAServices = require('fabric-ca-client');
const { Wallets, Gateway } = require('fabric-network');
const winston = require('winston');
const { DefaultQueryHandlerStrategies, DefaultEventHandlerStrategies } = require('fabric-network');

// Configure logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'identity-manager' },
  transports: [
    new winston.transports.File({ filename: path.join(__dirname, './logs/identity-error.log'), level: 'error' }),
    new winston.transports.File({ filename: path.join(__dirname, './logs/identity-combined.log') }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Configuration from environment variables
const CONNECTION_PROFILE_PATH = process.env.CONNECTION_PROFILE_PATH || path.resolve(__dirname, './connection-profile.json');
const WALLET_PATH = process.env.WALLET_PATH || path.join(__dirname, './wallet');
const MSP_ID = process.env.MSP_ID || 'Org1MSP';
const CA_URL = process.env.CA_URL || 'https://localhost:7054';
const ADMIN_IDENTITY = process.env.ADMIN_IDENTITY || 'admin';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'adminpw';
const ADMIN_MSP_PATH = process.env.ADMIN_MSP_PATH || '/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp';

/**
 * Initialize the wallet directory
 */
async function initWallet() {
  try {
    // Create wallet directory if it doesn't exist
    if (!fs.existsSync(WALLET_PATH)) {
      fs.mkdirSync(WALLET_PATH, { recursive: true });
      logger.info(`Created wallet directory at ${WALLET_PATH}`);
    }
    
    // Create a new file system based wallet for managing identities
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    logger.info(`Wallet initialized at ${WALLET_PATH}`);
    return wallet;
  } catch (error) {
    logger.error(`Failed to initialize wallet: ${error.message}`);
    throw error;
  }
}

/**
 * Check if an identity exists in the wallet
 * @param {string} identityLabel - The label of the identity to check
 * @returns {Promise<boolean>} - Whether the identity exists
 */
async function checkIdentityExists(identityLabel) {
  try {
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    const identity = await wallet.get(identityLabel);
    return identity !== undefined;
  } catch (error) {
    logger.error(`Error checking identity ${identityLabel}: ${error.message}`);
    return false;
  }
}

/**
 * Enroll the admin user from MSP materials
 * @returns {Promise<void>}
 */
async function enrollAdminFromMSP() {
  try {
    logger.info('Starting admin enrollment from MSP materials');
    
    // Check if admin identity already exists
    const wallet = await initWallet();
    const adminExists = await checkIdentityExists(ADMIN_IDENTITY);
    
    if (adminExists) {
      logger.info(`An identity for admin user "${ADMIN_IDENTITY}" already exists in the wallet`);
      return;
    }
    
    // Check if MSP path exists
    if (!fs.existsSync(ADMIN_MSP_PATH)) {
      throw new Error(`MSP path not found: ${ADMIN_MSP_PATH}`);
    }
    
    // Read the certificate and private key
    const certPath = path.join(ADMIN_MSP_PATH, 'signcerts', 'cert.pem');
    const keyDir = path.join(ADMIN_MSP_PATH, 'keystore');
    
    logger.info(`Looking for key files in: ${keyDir}`);
    const keyFiles = fs.readdirSync(keyDir);
    
    if (keyFiles.length === 0) {
      throw new Error('No key files found in the keystore directory');
    }
    
    logger.info(`Found keystore files: ${keyFiles.join(', ')}`);
    const keyPath = path.join(keyDir, keyFiles[0]);
    
    // Read the certificate and key
    const cert = fs.readFileSync(certPath).toString();
    const key = fs.readFileSync(keyPath).toString();
    
    // Create the identity
    const x509Identity = {
      credentials: {
        certificate: cert,
        privateKey: key,
      },
      mspId: MSP_ID,
      type: 'X.509',
    };
    
    // Store the identity in the wallet
    await wallet.put(ADMIN_IDENTITY, x509Identity);
    logger.info(`Successfully enrolled admin user "${ADMIN_IDENTITY}" from MSP and imported it into the wallet`);
    
  } catch (error) {
    logger.error(`Failed to enroll admin user from MSP: ${error.message}`);
    throw error;
  }
}

/**
 * Enroll the admin user with a CA
 * @returns {Promise<void>}
 */
async function enrollAdminWithCA() {
  try {
    logger.info('Starting admin enrollment with CA');
    
    // Check if admin identity already exists
    const wallet = await initWallet();
    const adminExists = await checkIdentityExists(ADMIN_IDENTITY);
    
    if (adminExists) {
      logger.info(`An identity for admin user "${ADMIN_IDENTITY}" already exists in the wallet`);
      return;
    }
    
    // Load the connection profile
    const connectionProfile = JSON.parse(fs.readFileSync(CONNECTION_PROFILE_PATH, 'utf8'));
    
    // Create a new CA client for interacting with the CA
    const caInfo = connectionProfile.certificateAuthorities['ca.org1.example.com'];
    const caTLSCACerts = caInfo.tlsCACerts.pem;
    const ca = new FabricCAServices(caInfo.url, { trustedRoots: caTLSCACerts, verify: false }, caInfo.caName);
    
    // Enroll the admin user with the CA
    const enrollment = await ca.enroll({ 
      enrollmentID: ADMIN_IDENTITY, 
      enrollmentSecret: ADMIN_PASSWORD 
    });
    
    // Create the identity
    const x509Identity = {
      credentials: {
        certificate: enrollment.certificate,
        privateKey: enrollment.key.toBytes(),
      },
      mspId: MSP_ID,
      type: 'X.509',
    };
    
    // Store the identity in the wallet
    await wallet.put(ADMIN_IDENTITY, x509Identity);
    logger.info(`Successfully enrolled admin user "${ADMIN_IDENTITY}" with CA and imported it into the wallet`);
    
  } catch (error) {
    logger.error(`Failed to enroll admin user with CA: ${error.message}`);
    throw error;
  }
}

/**
 * Register and enroll a new user
 * @param {string} userId - The ID of the user to register
 * @param {string} userRole - The role of the user
 * @param {string} affiliation - The affiliation of the user
 * @returns {Promise<void>}
 */
async function registerUser(userId, userRole = 'client', affiliation = 'org1.department1') {
  if (!userId) {
    throw new Error('User ID is required');
  }
  
  try {
    logger.info(`Starting registration for user ${userId}`);
    
    // Check if user identity already exists
    const wallet = await initWallet();
    const userExists = await checkIdentityExists(userId);
    
    if (userExists) {
      logger.info(`An identity for user "${userId}" already exists in the wallet`);
      return;
    }
    
    // Check if admin identity exists
    const adminExists = await checkIdentityExists(ADMIN_IDENTITY);
    if (!adminExists) {
      throw new Error(`Admin identity "${ADMIN_IDENTITY}" doesn't exist in the wallet. Please enroll admin first.`);
    }
    
    // Load the connection profile
    const connectionProfile = JSON.parse(fs.readFileSync(CONNECTION_PROFILE_PATH, 'utf8'));
    
    // Create a new CA client for interacting with the CA
    const caInfo = connectionProfile.certificateAuthorities['ca.org1.example.com'];
    const caTLSCACerts = caInfo.tlsCACerts.pem;
    const ca = new FabricCAServices(caInfo.url, { trustedRoots: caTLSCACerts, verify: false }, caInfo.caName);
    
    // Create a new gateway for connecting to our peer node
    const gateway = new Gateway();
    await gateway.connect(connectionProfile, { 
      wallet, 
      identity: ADMIN_IDENTITY, 
      discovery: { enabled: false } 
    });
    
    // Get the CA client object from the gateway
    const adminIdentity = await wallet.get(ADMIN_IDENTITY);
    const provider = wallet.getProviderRegistry().getProvider(adminIdentity.type);
    const adminUser = await provider.getUserContext(adminIdentity, ADMIN_IDENTITY);
    
    // Register the user
    const secret = await ca.register({
      affiliation,
      enrollmentID: userId,
      role: userRole
    }, adminUser);
    
    logger.info(`User ${userId} registered successfully`);
    
    // Enroll the user with the CA
    const enrollment = await ca.enroll({
      enrollmentID: userId,
      enrollmentSecret: secret
    });
    
    // Create the user identity
    const x509Identity = {
      credentials: {
        certificate: enrollment.certificate,
        privateKey: enrollment.key.toBytes(),
      },
      mspId: MSP_ID,
      type: 'X.509',
    };
    
    // Store the user identity in the wallet
    await wallet.put(userId, x509Identity);
    logger.info(`User ${userId} enrolled successfully and identity imported to wallet`);
    
    // Disconnect from the gateway
    gateway.disconnect();
    
  } catch (error) {
    logger.error(`Failed to register user ${userId}: ${error.message}`);
    throw error;
  }
}

/**
 * Create a service identity for automated processes
 * @param {string} serviceId - The ID for the service identity
 * @returns {Promise<void>}
 */
async function createServiceIdentity(serviceId) {
  try {
    // Register with special attributes for services
    await registerUser(serviceId, 'client', 'org1.department1');
    logger.info(`Service identity ${serviceId} created successfully`);
  } catch (error) {
    logger.error(`Failed to create service identity ${serviceId}: ${error.message}`);
    throw error;
  }
}

/**
 * List all identities in the wallet
 * @returns {Promise<Array>} - Array of identity labels
 */
async function listIdentities() {
  try {
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    const identityLabels = await wallet.list();
    logger.info(`Found ${identityLabels.length} identities in wallet`);
    return identityLabels;
  } catch (error) {
    logger.error(`Failed to list identities: ${error.message}`);
    throw error;
  }
}

/**
 * Delete an identity from the wallet
 * @param {string} identityLabel - The label of the identity to delete
 * @returns {Promise<boolean>} - Whether the deletion was successful
 */
async function deleteIdentity(identityLabel) {
  try {
    // Don't allow deletion of admin identity
    if (identityLabel === ADMIN_IDENTITY) {
      logger.warn(`Cannot delete admin identity "${ADMIN_IDENTITY}"`);
      return false;
    }
    
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    await wallet.remove(identityLabel);
    logger.info(`Identity ${identityLabel} deleted successfully`);
    return true;
  } catch (error) {
    logger.error(`Failed to delete identity ${identityLabel}: ${error.message}`);
    return false;
  }
}

/**
 * Verify an identity works correctly with the network
 * @param {string} identityLabel - The label of the identity to verify
 * @returns {Promise<boolean>} - Whether the identity is valid
 */
async function verifyIdentity(identityLabel) {
  try {
    logger.info(`Verifying identity ${identityLabel}`);
    
    // Check if identity exists
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    const identity = await wallet.get(identityLabel);
    
    if (!identity) {
      logger.warn(`Identity ${identityLabel} not found in wallet`);
      return false;
    }
    
    // Load the connection profile
    const connectionProfile = JSON.parse(fs.readFileSync(CONNECTION_PROFILE_PATH, 'utf8'));
    
    // Create a new gateway with this identity
    const gateway = new Gateway();
    await gateway.connect(connectionProfile, {
      wallet,
      identity: identityLabel,
      discovery: { enabled: false }
    });
    
    // Try to access the network - this will throw if identity is invalid
    await gateway.getNetwork(process.env.CHANNEL_NAME || 'neurashield-channel');
    
    // Disconnect from the gateway
    gateway.disconnect();
    
    logger.info(`Identity ${identityLabel} verified successfully`);
    return true;
  } catch (error) {
    logger.error(`Failed to verify identity ${identityLabel}: ${error.message}`);
    return false;
  }
}

/**
 * Main function to initialize all required identities
 */
async function initializeIdentities() {
  try {
    logger.info('Starting identity initialization');
    
    // Try enrolling admin via MSP first
    try {
      await enrollAdminFromMSP();
    } catch (mspError) {
      logger.warn(`MSP enrollment failed: ${mspError.message}. Trying CA enrollment...`);
      
      // Fall back to CA enrollment
      try {
        await enrollAdminWithCA();
      } catch (caError) {
        logger.error(`CA enrollment also failed: ${caError.message}`);
        throw new Error('Failed to enroll admin using both MSP and CA methods');
      }
    }
    
    // Create blockchain service identity if needed
    const serviceId = 'blockchain-service';
    const serviceExists = await checkIdentityExists(serviceId);
    
    if (!serviceExists) {
      await createServiceIdentity(serviceId);
    } else {
      logger.info(`Service identity ${serviceId} already exists`);
    }
    
    // Verify the admin identity works
    const adminValid = await verifyIdentity(ADMIN_IDENTITY);
    if (!adminValid) {
      throw new Error('Admin identity verification failed');
    }
    
    logger.info('Identity initialization completed successfully');
    return true;
  } catch (error) {
    logger.error(`Identity initialization failed: ${error.message}`);
    return false;
  }
}

/**
 * Get a gateway connection with the specified identity
 * @param {string} identityLabel - Identity to use for the connection
 * @param {Object} options - Additional connection options
 * @returns {Promise<Object>} - Gateway connection
 */
async function getGatewayConnection(identityLabel = ADMIN_IDENTITY, options = {}) {
  try {
    // Check if identity exists
    const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
    const identity = await wallet.get(identityLabel);
    
    if (!identity) {
      throw new Error(`Identity ${identityLabel} not found in wallet`);
    }
    
    // Load the connection profile
    const connectionProfile = JSON.parse(fs.readFileSync(CONNECTION_PROFILE_PATH, 'utf8'));
    
    // Create a new gateway
    const gateway = new Gateway();
    await gateway.connect(connectionProfile, {
      wallet,
      identity: identityLabel,
      discovery: { enabled: false },
      eventHandlerOptions: {
        commitTimeout: 300
      },
      ...options
    });
    
    logger.info(`Gateway connection established with identity ${identityLabel}`);
    return gateway;
  } catch (error) {
    logger.error(`Failed to get gateway connection: ${error.message}`);
    throw error;
  }
}

// Export functions
module.exports = {
  initializeIdentities,
  enrollAdminFromMSP,
  enrollAdminWithCA,
  registerUser,
  createServiceIdentity,
  listIdentities,
  deleteIdentity,
  verifyIdentity,
  getGatewayConnection,
  checkIdentityExists
}; 