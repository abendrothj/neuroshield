/**
 * NeuraShield Backend Server
 */

require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { Gateway, Wallets } = require('fabric-network');
const { create } = require('ipfs-http-client');
const axios = require('axios');
const rateLimit = require('express-rate-limit');
const { body, param, validationResult } = require('express-validator');
const winston = require('winston');
const { metricsApp, httpRequestDuration, httpRequestsTotal, blockchainSyncStatus, modelAccuracy } = require('./metrics');
const bodyParser = require('body-parser');
const blockchainIntegration = require('./blockchain-integration');
const identityManager = require('../identity-manager');
const { initObservability, logger } = require('./local-observability');
const observability = initObservability();

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, '../logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

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
    defaultMeta: { service: 'neurashield-backend' },
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ 
            filename: 'logs/error.log', 
            level: 'error',
            // Don't log sensitive data
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.json(),
                winston.format(info => {
                    if (info.password || info.token || info.secret) {
                        info.password = info.password ? '[REDACTED]' : undefined;
                        info.token = info.token ? '[REDACTED]' : undefined;
                        info.secret = info.secret ? '[REDACTED]' : undefined;
                    }
                    return info;
                })()
            )
        }),
        new winston.transports.File({ filename: 'logs/combined.log' })
    ],
    // Handle uncaught exceptions but don't expose them in production
    exceptionHandlers: [
        new winston.transports.File({ filename: 'logs/exceptions.log' })
    ]
});

if (process.env.NODE_ENV !== 'production') {
    logger.add(new winston.transports.Console({
        format: winston.format.simple()
    }));
}

const app = express();

// Apply security middleware
app.use(helmet());
app.use(cors());
app.use(bodyParser.json());

// Apply observability middleware
app.use(observability.middleware.metrics);

// Add rate limiting
const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    standardHeaders: true,
    legacyHeaders: false,
    message: 'Too many requests from this IP, please try again after 15 minutes'
});

// Apply rate limiting to all API routes
app.use('/api', apiLimiter);

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

// API routes
app.use('/api', require('./routes/api')(observability));

// Metrics endpoint
app.get('/metrics', (req, res) => {
  const metrics = {
    service: process.env.SERVICE_NAME || 'neurashield-backend',
    version: process.env.SERVICE_VERSION || '1.0.0',
    uptime: process.uptime(),
    timestamp: new Date(),
    memoryUsage: process.memoryUsage(),
    counters: {},
    histograms: {}
  };
  
  // Add counter metrics
  observability.metricsTracker.counters.forEach((counter, name) => {
    metrics.counters[name] = {
      value: counter.value,
      description: counter.description
    };
  });
  
  // Add histogram metrics
  observability.metricsTracker.histograms.forEach((histogram, name) => {
    metrics.histograms[name] = {
      description: histogram.description,
      stats: histogram.getStats()
    };
  });
  
  res.status(200).json(metrics);
});

// Error handling middleware
app.use(observability.middleware.errorReporting);
app.use((err, req, res, next) => {
  logger.error('Unhandled error', { 
    error: { 
      message: err.message,
      stack: err.stack
    }
  });
  
  res.status(500).json({
    error: {
      message: 'An internal server error occurred',
      code: 'INTERNAL_ERROR'
    }
  });
});

// Start server
const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => {
  logger.info(`Server listening on port ${PORT}`);
  logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM signal received. Shutting down gracefully.');
  server.close(() => {
    logger.info('HTTP server closed.');
    process.exit(0);
  });
});

module.exports = app; 