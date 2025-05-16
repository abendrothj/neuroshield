#!/usr/bin/env node

/**
 * NeuraShield Blockchain Event Verification Script
 * 
 * This script is used by the Python API to verify blockchain events.
 * It calls the blockchain adapter and returns the verification certificate.
 */

const path = require('path');

// Get the event ID from command line arguments
const eventId = process.argv[2];

if (!eventId) {
  console.error('Error: No event ID provided');
  console.error('Usage: node verify-blockchain-event.js <event-id>');
  process.exit(1);
}

// Import the blockchain adapter
const { getBlockchainImplementation } = require('../backend/src/blockchain-adapter');

async function verifyEvent() {
  try {
    // Get blockchain implementation
    const blockchain = await getBlockchainImplementation();
    
    // Initialize the blockchain connection
    await blockchain.initBlockchain();
    
    // Generate verification certificate
    const certificate = await blockchain.generateVerificationCertificate(eventId);
    
    // Output the certificate as JSON
    console.log(JSON.stringify(certificate));
    
    process.exit(0);
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

// Run the verification
verifyEvent().catch(error => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
}); 