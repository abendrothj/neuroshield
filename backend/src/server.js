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
app.use(limiter);

// IPFS configuration
const ipfs = create({ url: process.env.IPFS_URL || 'http://localhost:5001' });

// Fabric configuration
const ccpPath = path.resolve(__dirname, '..', 'connection-profile.json');
const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

// Initialize Fabric gateway
async function getGateway() {
    const wallet = await Wallets.newFileSystemWallet(path.join(__dirname, 'wallet'));
    const gateway = new Gateway();
    await gateway.connect(ccp, { wallet, identity: 'admin', discovery: { enabled: true, asLocalhost: true } });
    return gateway;
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
    res.status(200).json({ status: 'healthy' });
});

// Metrics endpoint
app.use('/metrics', metricsApp);

// AI service integration
app.post('/api/analyze', 
    [
        body('data').isObject().withMessage('Data must be an object'),
        validate
    ],
    async (req, res) => {
        try {
            const { data } = req.body;
            const aiResponse = await axios.post(`${process.env.AI_SERVICE_URL}/analyze`, { data });
            res.json(aiResponse.data);
        } catch (error) {
            logger.error(`Failed to analyze data: ${error}`);
            res.status(500).json({ error: error.message });
        }
    }
);

// Routes
app.post('/api/events', 
    [
        body('id').isString().withMessage('ID must be a string'),
        body('timestamp').isISO8601().withMessage('Timestamp must be a valid ISO8601 date'),
        body('type').isString().withMessage('Type must be a string'),
        body('details').isObject().withMessage('Details must be an object'),
        validate
    ],
    async (req, res) => {
        try {
            const { id, timestamp, type, details } = req.body;
            
            // Upload details to IPFS
            const ipfsResult = await ipfs.add(JSON.stringify(details));
            const ipfsHash = ipfsResult.path;

            // Store event on Fabric
            const gateway = await getGateway();
            const network = await gateway.getNetwork('neurashield-channel');
            const contract = network.getContract('neurashield');

            await contract.submitTransaction('LogEvent', id, timestamp, type, details, ipfsHash);
            
            logger.info(`Event logged successfully: ${id}`);
            res.json({ success: true, message: 'Event logged successfully', ipfsHash });
        } catch (error) {
            logger.error(`Failed to log event: ${error}`);
            res.status(500).json({ error: error.message });
        }
    }
);

app.get('/api/events/:id', 
    [
        param('id').isString().withMessage('ID must be a string'),
        validate
    ],
    async (req, res) => {
        try {
            const gateway = await getGateway();
            const network = await gateway.getNetwork('neurashield-channel');
            const contract = network.getContract('neurashield');

            const result = await contract.evaluateTransaction('QueryEvent', req.params.id);
            res.json(JSON.parse(result.toString()));
        } catch (error) {
            logger.error(`Failed to query event: ${error}`);
            res.status(500).json({ error: error.message });
        }
    }
);

app.get('/api/events', async (req, res) => {
    try {
        const gateway = await getGateway();
        const network = await gateway.getNetwork('neurashield-channel');
        const contract = network.getContract('neurashield');

        const result = await contract.evaluateTransaction('QueryAllEvents');
        res.json(JSON.parse(result.toString()));
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