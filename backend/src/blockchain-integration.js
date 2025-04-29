/**
 * NeuraShield Blockchain Integration Service
 * 
 * This service integrates the system with blockchain for
 * secure, tamper-proof audit logs of security events.
 */

const { getBlockchainImplementation } = require('./blockchain-adapter');
const express = require('express');
const winston = require('winston');
const path = require('path');

// Setup logging
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ level, message, timestamp }) => {
            return `${timestamp} ${level}: ${message}`;
        })
    ),
    defaultMeta: { service: 'blockchain-integration' },
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ 
            filename: path.join(__dirname, '..', 'logs/blockchain-integration.log')
        })
    ]
});

// Initialize blockchain implementation
let blockchainImpl = null;

/**
 * Initialize the blockchain integration
 */
async function initializeBlockchain() {
    try {
        logger.info('Initializing blockchain integration service');
        blockchainImpl = await getBlockchainImplementation();
        await blockchainImpl.initBlockchain();
        logger.info('Blockchain integration service initialized successfully');
        return true;
    } catch (error) {
        logger.error(`Failed to initialize blockchain integration: ${error.message}`);
        throw error;
    }
}

/**
 * Process a security event
 * @param {Object} eventData - Event data to record on blockchain
 */
async function processSecurityEvent(eventData) {
    try {
        if (!blockchainImpl) {
            await initializeBlockchain();
        }
        
        return await blockchainImpl.processSecurityEvent(eventData);
    } catch (error) {
        logger.error(`Error processing security event: ${error.message}`);
        throw error;
    }
}

/**
 * Fetch all events from the blockchain
 */
async function fetchEvents() {
    try {
        if (!blockchainImpl) {
            await initializeBlockchain();
        }
        
        return await blockchainImpl.fetchEvents();
    } catch (error) {
        logger.error(`Error fetching events: ${error.message}`);
        throw error;
    }
}

/**
 * Fetch a specific event by ID
 * @param {string} eventId - ID of the event to fetch
 */
async function fetchEvent(eventId) {
    try {
        if (!blockchainImpl) {
            await initializeBlockchain();
        }
        
        return await blockchainImpl.fetchEvent(eventId);
    } catch (error) {
        logger.error(`Error fetching event ${eventId}: ${error.message}`);
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
        if (!blockchainImpl) {
            await initializeBlockchain();
        }
        
        await blockchainImpl.aiDetectionWebhook(req, res);
    } catch (error) {
        logger.error(`Error in AI detection webhook: ${error.message}`);
        res.status(500).json({ error: error.message });
    }
}

module.exports = {
    initializeBlockchain,
    processSecurityEvent,
    fetchEvents,
    fetchEvent,
    aiDetectionWebhook
}; 