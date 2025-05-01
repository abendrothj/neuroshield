const express = require('express');
const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');
const helmet = require('helmet');
const winston = require('winston');
const cors = require('cors');
require('dotenv').config();

// Setup production logging
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        process.env.NODE_ENV === 'production' 
            ? winston.format.json()
            : winston.format.printf(({ level, message, timestamp }) => {
                return `${timestamp} ${level}: ${message}`;
            })
    ),
    defaultMeta: { service: 'neurashield-blockchain' },
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ 
            filename: 'logs/error.log', 
            level: 'error'
        }),
        new winston.transports.File({ filename: 'logs/combined.log' })
    ]
});

// Configuration
const PORT = process.env.PORT || 8080;
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'neurashield-channel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'neurashield';
const NETWORK_CONFIG_PATH = process.env.NETWORK_CONFIG_PATH || '/app/config/connection-profile.json';
const WALLET_PATH = process.env.WALLET_PATH || '/app/wallet';
const USER_ID = process.env.USER_ID || 'admin';

const app = express();

// Middleware
app.use(helmet());
app.use(express.json());
app.use(cors());

// Function to get connection to the blockchain
async function getGateway() {
    try {
        // Check if network config exists
        if (!fs.existsSync(NETWORK_CONFIG_PATH)) {
            throw new Error(`Network config not found at ${NETWORK_CONFIG_PATH}`);
        }

        // Load the network configuration
        const ccp = JSON.parse(fs.readFileSync(NETWORK_CONFIG_PATH, 'utf8'));
        
        // Create a file system wallet
        const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
        
        // Check if the user identity exists in the wallet
        const identity = await wallet.get(USER_ID);
        if (!identity) {
            throw new Error(`Identity for ${USER_ID} not found in wallet`);
        }
        
        // Create a new gateway for connecting to the peer node
        const gateway = new Gateway();
        await gateway.connect(ccp, { 
            wallet, 
            identity: USER_ID, 
            discovery: { enabled: true, asLocalhost: false }
        });
        
        return gateway;
    } catch (error) {
        logger.error(`Failed to connect to gateway: ${error.message}`);
        throw error;
    }
}

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString()
    });
});

// Record threat to blockchain
app.post('/api/recordThreat', async (req, res) => {
    const { cid, threatData } = req.body;
    
    if (!cid || !threatData) {
        return res.status(400).json({
            error: 'Missing required parameters',
            message: 'Both cid and threatData are required'
        });
    }
    
    let gateway;
    try {
        // Get connection to blockchain
        gateway = await getGateway();
        
        // Get the network and contract
        const network = await gateway.getNetwork(CHANNEL_NAME);
        const contract = network.getContract(CHAINCODE_NAME);
        
        // Submit transaction to record the threat
        await contract.submitTransaction(
            'recordThreat', 
            cid, 
            JSON.stringify(threatData)
        );
        
        logger.info(`Successfully recorded threat to blockchain, IPFS CID: ${cid}`);
        res.status(200).json({
            status: 'success',
            cid: cid,
            message: 'Threat recorded to blockchain'
        });
    } catch (error) {
        logger.error(`Failed to record threat to blockchain: ${error.message}`);
        res.status(500).json({
            error: 'Failed to record threat',
            message: error.message
        });
    } finally {
        if (gateway) {
            try {
                gateway.disconnect();
            } catch (error) {
                logger.error(`Error disconnecting from gateway: ${error.message}`);
            }
        }
    }
});

// Get threat data from blockchain
app.get('/api/getThreat/:cid', async (req, res) => {
    const { cid } = req.params;
    
    if (!cid) {
        return res.status(400).json({
            error: 'Missing required parameter',
            message: 'CID is required'
        });
    }
    
    let gateway;
    try {
        // Get connection to blockchain
        gateway = await getGateway();
        
        // Get the network and contract
        const network = await gateway.getNetwork(CHANNEL_NAME);
        const contract = network.getContract(CHAINCODE_NAME);
        
        // Evaluate transaction to get the threat
        const threatData = await contract.evaluateTransaction('getThreat', cid);
        
        // Parse the threat data
        const threat = JSON.parse(threatData.toString());
        
        logger.info(`Successfully retrieved threat data for CID: ${cid}`);
        res.status(200).json({
            status: 'success',
            cid: cid,
            data: threat
        });
    } catch (error) {
        logger.error(`Failed to get threat data from blockchain: ${error.message}`);
        res.status(500).json({
            error: 'Failed to get threat data',
            message: error.message
        });
    } finally {
        if (gateway) {
            try {
                gateway.disconnect();
            } catch (error) {
                logger.error(`Error disconnecting from gateway: ${error.message}`);
            }
        }
    }
});

// Start the server
app.listen(PORT, () => {
    logger.info(`Blockchain service running on port ${PORT}`);
}); 