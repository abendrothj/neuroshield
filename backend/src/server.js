const express = require('express');
const { Gateway, Wallets } = require('fabric-network');
const { create } = require('ipfs-http-client');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const rateLimit = require('express-rate-limit');
const { body, param, validationResult } = require('express-validator');
const winston = require('winston');
const { metricsApp, httpRequestDuration, httpRequestsTotal, blockchainSyncStatus, modelAccuracy } = require('./metrics');
require('dotenv').config();

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'error.log', level: 'error' }),
        new winston.transports.File({ filename: 'combined.log' })
    ]
});

if (process.env.NODE_ENV !== 'production') {
    logger.add(new winston.transports.Console({
        format: winston.format.simple()
    }));
}

const app = express();
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});

// Specific rate limiters for different operations
const analyzeRateLimiter = rateLimit({
    windowMs: 5 * 60 * 1000, // 5 minutes
    max: 20, // limit each IP to 20 requests per windowMs
    message: 'Too many analysis requests, please try again later'
});

const metricsRateLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 30, // limit each IP to 30 requests per windowMs
    message: 'Too many metrics requests, please try again later'
});

const eventsRateLimiter = rateLimit({
    windowMs: 10 * 60 * 1000, // 10 minutes
    max: 50, // limit each IP to 50 requests per windowMs
    message: 'Too many event requests, please try again later'
});

const trainingRateLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 60 minutes
    max: 5, // limit each IP to 5 requests per hour
    message: 'Too many training requests, please try again later'
});

app.use(limiter); // Apply general rate limiting to all endpoints not specifically limited

// IPFS configuration
const ipfs = create({ url: process.env.IPFS_URL || 'http://localhost:5001' });

// Fabric configuration
const ccpPath = path.resolve(__dirname, '..', 'connection-profile.json');
const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

// Connection pool for Fabric gateway
const gatewayPool = {
    pool: [], // Store active gateways
    maxSize: 5, // Maximum number of gateways in the pool
    activeConnections: 0,
    
    async getGateway() {
        // If a gateway is available in the pool, return it
        if (this.pool.length > 0) {
            logger.debug('Reusing gateway from pool');
            this.activeConnections++;
            return this.pool.pop();
        }
        
        // Otherwise create a new gateway
        logger.debug('Creating new gateway');
        const wallet = await Wallets.newFileSystemWallet(path.join(__dirname, '..', 'wallet'));
        const gateway = new Gateway();
        await gateway.connect(ccp, { 
            wallet, 
            identity: process.env.USER_ID || 'admin', 
            discovery: { enabled: true, asLocalhost: true }
        });
        this.activeConnections++;
        return gateway;
    },
    
    releaseGateway(gateway) {
        // Only add gateway back to pool if not exceeding max size
        try {
            if (this.pool.length < this.maxSize) {
                this.pool.push(gateway);
                logger.debug('Gateway returned to pool');
            } else {
                // If pool is full, disconnect this gateway
                gateway.disconnect();
                logger.debug('Gateway disconnected (pool full)');
            }
        } catch (error) {
            logger.error(`Error releasing gateway: ${error}`);
            try {
                gateway.disconnect();
            } catch (e) {
                logger.error(`Error disconnecting gateway: ${e}`);
            }
        }
        this.activeConnections--;
    }
};

// Helper function to get and safely release a gateway
async function withGateway(callback) {
    let gateway = null;
    try {
        gateway = await gatewayPool.getGateway();
        return await callback(gateway);
    } finally {
        if (gateway) {
            gatewayPool.releaseGateway(gateway);
        }
    }
}

// Initialize Fabric gateway - kept for backward compatibility
async function getGateway() {
    return await gatewayPool.getGateway();
}

// Error handling middleware
const errorHandler = (err, req, res, next) => {
    logger.error('Error:', err);
    res.status(500).json({
        error: 'Internal Server Error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'An error occurred'
    });
};

// Input validation middleware
const validate = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
    }
    next();
};

// Metrics middleware
const metricsMiddleware = (req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = (Date.now() - start) / 1000;
        httpRequestDuration
            .labels(req.method, req.route?.path || req.path, res.statusCode)
            .observe(duration);
        httpRequestsTotal
            .labels(req.method, req.route?.path || req.path, res.statusCode)
            .inc();
    });
    next();
};

app.use(metricsMiddleware);

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({ 
        status: 'healthy',
        gatewayPool: {
            poolSize: gatewayPool.pool.length,
            activeConnections: gatewayPool.activeConnections
        }
    });
});

// Metrics endpoint
app.use('/metrics', metricsApp);

// Update blockchain sync status periodically
const updateBlockchainStatus = async () => {
    let gateway;
    try {
        gateway = await getGateway();
        const network = await gateway.getNetwork('neurashield-channel');
        
        // Basic check if the blockchain is accessible and responding
        if (network) {
            blockchainSyncStatus.set(1); // 1 = synced
            logger.info('Blockchain sync status: synced');
        } else {
            blockchainSyncStatus.set(0); // 0 = not synced
            logger.warn('Blockchain sync status: not synced');
        }
    } catch (error) {
        blockchainSyncStatus.set(0); // 0 = not synced
        logger.error(`Error checking blockchain sync status: ${error}`);
    } finally {
        if (gateway) {
            try {
                gateway.disconnect();
            } catch (err) {
                logger.error(`Error disconnecting gateway: ${err}`);
            }
        }
    }
};

// Run immediately and then every 5 minutes
updateBlockchainStatus();
setInterval(updateBlockchainStatus, 5 * 60 * 1000);

// Helper function to record threat data to blockchain
async function recordToBlockchain(cid, results) {
    try {
        return await withGateway(async (gateway) => {
            const network = await gateway.getNetwork(process.env.CHANNEL_NAME);
            const contract = network.getContract(process.env.CONTRACT_NAME);
            
            // Create a summary of the threats
            const threatSummary = results.map(r => ({
                level: r.threat_level,
                confidence: r.confidence,
                timestamp: new Date().toISOString()
            }));
            
            // Submit transaction to record the threat
            await contract.submitTransaction(
                'recordThreat', 
                cid, 
                JSON.stringify(threatSummary)
            );
            
            logger.info(`Successfully recorded threat to blockchain, IPFS CID: ${cid}`);
            return true;
        });
    } catch (error) {
        logger.error(`Blockchain recording error: ${error}`);
        throw error;
    }
}

// AI service integration
app.post('/api/analyze', 
    [
        analyzeRateLimiter,
        body('data').isArray().withMessage('Data must be an array of objects'),
        validate
    ],
    async (req, res) => {
        try {
            const { data } = req.body;
            
            // Validate data
            if (!data || !Array.isArray(data) || data.length === 0) {
                return res.status(400).json({ 
                    error: 'Invalid request format',
                    message: 'The data field must be a non-empty array of objects'
                });
            }
            
            logger.info(`Sending data to AI service at ${process.env.AI_SERVICE_URL}/analyze`);
            const startTime = Date.now();
            
            // Send to AI service for analysis
            const aiResponse = await axios.post(`${process.env.AI_SERVICE_URL}/analyze`, { data });
            const processingTime = (Date.now() - startTime) / 1000;
            
            // Update model accuracy if provided in the response
            if (aiResponse.data.model_info && aiResponse.data.model_info.accuracy) {
                modelAccuracy.set(aiResponse.data.model_info.accuracy);
            }
            
            // Extract the threat results
            const results = aiResponse.data.results;
            
            // Record threats to blockchain and IPFS if needed
            if (results.some(result => result.threat_level !== "Normal")) {
                let cid = null;
                
                // Try to store on IPFS, but continue if it fails
                try {
                    // Store the raw data on IPFS
                    const ipfsResult = await ipfs.add(JSON.stringify({
                        data,
                        analysis: results,
                        timestamp: new Date().toISOString()
                    }));
                    
                    cid = ipfsResult.cid.toString();
                    logger.info(`Stored threat data on IPFS with CID: ${cid}`);
                    
                    // Include the IPFS reference in the response
                    aiResponse.data.ipfs_cid = cid;
                } catch (ipfsError) {
                    logger.error(`Failed to store on IPFS: ${ipfsError}`);
                    // Create a fallback CID value to indicate IPFS storage failed
                    cid = 'ipfs-storage-failed-' + Date.now();
                    aiResponse.data.ipfs_storage_error = true;
                }
                
                // Record to blockchain (attempt only, don't block the response)
                // This is run asynchronously to not delay the response
                if (cid) {
                    recordToBlockchain(cid, results).catch(err => {
                        logger.error(`Failed to record to blockchain: ${err}`);
                    });
                }
            }
            
            logger.info(`Analysis completed in ${processingTime}s, returned ${results.length} result(s)`);
            res.json(aiResponse.data);
        } catch (error) {
            logger.error(`Failed to analyze data: ${error.message}`);
            res.status(500).json({ error: error.message });
        }
    }
);

// AI metrics endpoint for frontend
app.get('/api/ai-metrics', metricsRateLimiter, async (req, res) => {
    try {
        // Get metrics from AI service's formatted endpoint
        logger.info(`Fetching metrics from AI service at ${process.env.AI_SERVICE_URL}/api/metrics`);
        const aiResponse = await axios.get(`${process.env.AI_SERVICE_URL}/api/metrics`);
        
        // Return metrics from AI service
        const metrics = aiResponse.data;
        
        // Add additional context from the blockchain
        try {
            let gateway = await getGateway();
            const network = await gateway.getNetwork(process.env.CHANNEL_NAME);
            const contract = network.getContract(process.env.CONTRACT_NAME);
            
            // Get threat stats from blockchain
            const threatStatsBuffer = await contract.evaluateTransaction('getThreatStats');
            const threatStats = JSON.parse(threatStatsBuffer.toString());
            
            metrics.threat_stats = threatStats;
            gateway.disconnect();
        } catch (blockchainError) {
            logger.warn(`Could not get blockchain metrics: ${blockchainError.message}`);
            // Continue without blockchain metrics
        }
        
        res.json(metrics);
    } catch (error) {
        logger.error(`Failed to get AI metrics: ${error.message}`);
        // Return default values if can't get real metrics
        res.json({
            accuracy: modelAccuracy.get() || 0.9,
            inference_time: 0.05,
            memory_usage: 500 * 1024 * 1024,
            predictions_total: 10000,
            error_rate: 0.01,
            model_version: "1.0.0",
            threat_detection_rate: 0.05,
            gpu_utilization: 0.3
        });
    }
});

// Model training endpoint
app.post('/api/train-model',
    [
        trainingRateLimiter,
        body('epochs').isInt({ min: 1, max: 1000 }).withMessage('Epochs must be between 1 and 1000'),
        body('batchSize').isInt({ min: 1, max: 512 }).withMessage('Batch size must be between 1 and 512'),
        body('learningRate').isFloat({ min: 0.0001, max: 0.1 }).withMessage('Learning rate must be between 0.0001 and 0.1'),
        body('datasetSize').isInt({ min: 1000, max: 100000 }).withMessage('Dataset size must be between 1000 and 100000'),
        body('validationSplit').isFloat({ min: 0.1, max: 0.5 }).withMessage('Validation split must be between 0.1 and 0.5'),
        validate
    ],
    async (req, res) => {
        try {
            const { epochs, batchSize, learningRate, datasetSize, validationSplit } = req.body;
            
            logger.info(`Initiating model training with epochs=${epochs}, batchSize=${batchSize}, learningRate=${learningRate}`);
            
            // Call the AI service to start training
            const aiResponse = await axios.post(`${process.env.AI_SERVICE_URL}/train`, {
                epochs,
                batch_size: batchSize,
                learning_rate: learningRate,
                dataset_size: datasetSize,
                validation_split: validationSplit
            });
            
            res.json({
                success: true,
                message: 'Model training initiated successfully',
                jobId: aiResponse.data.job_id
            });
        } catch (error) {
            logger.error(`Failed to initiate model training: ${error}`);
            res.status(500).json({ success: false, message: error.message });
        }
    }
);

// Model training status endpoint
app.get('/api/train-model/:jobId', trainingRateLimiter, async (req, res) => {
    try {
        const { jobId } = req.params;
        
        // Call the AI service to get training status
        const aiResponse = await axios.get(`${process.env.AI_SERVICE_URL}/train/${jobId}`);
        
        res.json(aiResponse.data);
    } catch (error) {
        logger.error(`Failed to get training status: ${error}`);
        res.status(500).json({ success: false, message: error.message });
    }
});

// Routes
app.post('/api/events', 
    [
        eventsRateLimiter,
        body('id').isString().withMessage('ID must be a string'),
        body('timestamp').isISO8601().withMessage('Timestamp must be a valid ISO8601 date'),
        body('type').isString().withMessage('Type must be a string'),
        body('details').isObject().withMessage('Details must be an object'),
        validate
    ],
    async (req, res) => {
        try {
            const { id, timestamp, type, details } = req.body;
            
            // Upload details to IPFS with proper error handling
            let ipfsHash;
            try {
                const ipfsResult = await ipfs.add(JSON.stringify(details));
                ipfsHash = ipfsResult.path;
                logger.info(`Stored details on IPFS with hash: ${ipfsHash}`);
            } catch (ipfsError) {
                logger.error(`IPFS storage error: ${ipfsError}`);
                // Continue without IPFS storage, using a fallback
                ipfsHash = 'ipfs-storage-unavailable';
            }

            // Store event on Fabric
            await withGateway(async (gateway) => {
                const network = await gateway.getNetwork(process.env.CHANNEL_NAME);
                const contract = network.getContract(process.env.CONTRACT_NAME);
                
                await contract.submitTransaction('LogEvent', id, timestamp, type, JSON.stringify(details), ipfsHash);
            });
            
            logger.info(`Event logged successfully: ${id}`);
            res.json({ 
                success: true, 
                message: 'Event logged successfully', 
                ipfsHash,
                ipfs_storage_success: ipfsHash !== 'ipfs-storage-unavailable'
            });
        } catch (error) {
            logger.error(`Failed to log event: ${error}`);
            res.status(500).json({ error: error.message });
        }
    }
);

app.get('/api/events/:id', 
    [
        eventsRateLimiter,
        param('id').isString().withMessage('ID must be a string'),
        validate
    ],
    async (req, res) => {
        try {
            const result = await withGateway(async (gateway) => {
                const network = await gateway.getNetwork(process.env.CHANNEL_NAME);
                const contract = network.getContract(process.env.CONTRACT_NAME);
                
                const resultBuffer = await contract.evaluateTransaction('QueryEvent', req.params.id);
                return JSON.parse(resultBuffer.toString());
            });
            
            res.json(result);
        } catch (error) {
            logger.error(`Failed to query event: ${error}`);
            res.status(500).json({ error: error.message });
        }
    }
);

app.get('/api/events', eventsRateLimiter, async (req, res) => {
    try {
        const result = await withGateway(async (gateway) => {
            const network = await gateway.getNetwork(process.env.CHANNEL_NAME);
            const contract = network.getContract(process.env.CONTRACT_NAME);
            
            const resultBuffer = await contract.evaluateTransaction('QueryAllEvents');
            return JSON.parse(resultBuffer.toString());
        });
        
        res.json(result);
    } catch (error) {
        logger.error(`Failed to query all events: ${error}`);
        res.status(500).json({ error: error.message });
    }
});

// Apply error handling middleware
app.use(errorHandler);

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    logger.info(`Server running on port ${PORT}`);
}); 