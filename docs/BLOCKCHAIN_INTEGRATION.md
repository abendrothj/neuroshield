# NeuraShield Blockchain Integration

This document outlines how NeuraShield integrates with Hyperledger Fabric blockchain to provide immutable logging of security events detected by the AI system.

## Architecture Overview

NeuraShield uses a Hyperledger Fabric blockchain network to ensure the integrity and immutability of security event logs. The architecture consists of:

1. **Identity Management System**: Manages cryptographic identities for blockchain interactions
2. **Blockchain Integration Service**: Handles the submission of security events to the blockchain
3. **Smart Contract (Chaincode)**: Defines the business logic for storing and querying events
4. **IPFS Integration**: Optional storage for detailed event data with only hashes stored on-chain
5. **API Layer**: Exposes endpoints for blockchain interactions

## Components

### Identity Manager (`backend/identity-manager.js`)

The identity manager handles user identities for the Hyperledger Fabric blockchain:

- Creates and manages the file system wallet for storing cryptographic materials
- Enrolls admin users with Certificate Authority or from MSP materials
- Registers and enrolls new users
- Verifies identity validity
- Provides gateway connections to the blockchain network

Key configuration:
- `CONNECTION_PROFILE_PATH`: Path to connection profile JSON
- `WALLET_PATH`: Path to wallet directory
- `MSP_ID`: Membership Service Provider ID (default: 'Org1MSP')
- `CA_URL`: Certificate Authority URL
- `ADMIN_IDENTITY`: Admin identity label

### Blockchain Integration Service (`backend/src/blockchain-integration.js`)

The service acts as a bridge between the AI detection system and Hyperledger Fabric:

- Processes security events from the AI system
- Classifies events based on threat level
- Stores full event details in IPFS (if enabled)
- Submits event summaries to the blockchain using a queue with retry logic
- Exposes webhook endpoint for AI detections

Key features:
- Processing queue with retry capability
- Gateway connection pooling
- Optional IPFS storage for full event details
- Fallback mechanisms for service unavailability

### Smart Contract (`backend/chaincode/chaincode.go`)

Written in Go using the Fabric Contract API, the chaincode defines:

- **Data Model**: SecurityEvent struct with ID, timestamp, type, details, and IPFS hash
- **Transactions**:
  - `InitLedger`: Initializes the ledger with a genesis event
  - `LogEvent`: Records security events
  - `QueryEvent`: Retrieves specific events by ID
  - `QueryAllEvents`: Retrieves all events from the ledger

### Server Integration (`backend/src/server.js`)

The server exposes REST API endpoints for blockchain interaction:

- `/api/events`: POST for logging events, GET for retrieving all events
- `/api/events/:id`: GET for retrieving a specific event
- `/api/v1/ai-detection`: Webhook for AI detection events
- `/api/v1/events`: Alternative endpoint for querying blockchain events

Additional features:
- Connection pooling for Fabric gateways
- Periodic blockchain connectivity checks
- Metrics for monitoring blockchain sync status

## Data Flow

1. **Detection**: AI system detects potential security threats
2. **Event Creation**: System formats threat data as security events
3. **IPFS Storage** (optional): Full event details stored in IPFS, returning a content ID (CID)
4. **Blockchain Submission**: Event summary and IPFS hash submitted to blockchain
5. **Verification**: Transaction committed to ledger after consensus
6. **Querying**: Events can be retrieved through API endpoints

## Security Event Processing

```javascript
// Example of event processing flow
async function processSecurityEvent(threatData) {
  // Generate unique event ID
  const eventId = `event-${crypto.randomBytes(4).toString('hex')}-${Date.now()}`;
  
  // Format timestamp
  const timestamp = new Date().toISOString();
  
  // Classify event type based on threat level
  let eventType = calculateEventType(threatData.confidence);
  
  // Store full details in IPFS if enabled
  let ipfsHash = '';
  if (process.env.IPFS_ENABLED === 'true') {
    ipfsHash = await storeInIPFS(fullEventDetails);
  }
  
  // Submit to blockchain with retry capability
  return await submitToBlockchainWithRetry({
    id: eventId,
    timestamp: timestamp,
    type: eventType,
    details: eventSummary,
    ipfsHash: ipfsHash
  });
}
```

## Resilience Features

1. **Retry Logic**: Failed blockchain submissions are retried with exponential backoff
2. **Fallbacks**: System continues operating if blockchain or IPFS is unavailable
3. **Connection Pooling**: Efficient reuse of blockchain connections
4. **Status Monitoring**: Regular checks of blockchain connectivity

## Testing the Integration

1. **Environment Setup**:
   - Ensure Hyperledger Fabric network is running
   - Verify the connection profile is correctly configured
   - Check admin identity is properly enrolled

2. **Identity Verification**:
   - Test admin and service identities are valid
   - Verify wallet contains necessary cryptographic materials

3. **Event Submission**:
   - Create test security events
   - Submit to blockchain and verify transaction success
   - Query events to ensure data integrity

4. **IPFS Integration** (if enabled):
   - Verify event details are stored in IPFS
   - Confirm IPFS hash is correctly recorded on blockchain
   - Retrieve and validate full details from IPFS using hash

## Configuration

```
# Blockchain Configuration
CONNECTION_PROFILE_PATH=/home/jub/Cursor/neurashield/backend/connection-profile.json
CHANNEL_NAME=neurashield-channel
CHAINCODE_NAME=neurashield
WALLET_PATH=/home/jub/Cursor/neurashield/backend/wallet
MSP_ID=Org1MSP
BLOCKCHAIN_IDENTITY=admin

# IPFS Configuration
IPFS_ENABLED=true
IPFS_API=http://localhost:5001/api/v0
IPFS_GATEWAY=http://localhost:8080/ipfs/

# Resilience Configuration
EVENT_QUEUE_CONCURRENCY=5
MAX_RETRIES=5
RETRY_DELAY=5000 