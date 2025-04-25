# NeuraShield Blockchain Integration

This module provides the integration layer between the AI threat detection system and the Hyperledger Fabric blockchain for secure, immutable audit logging of security events.

## Overview

The blockchain integration component:

1. Receives threat detection events from the AI system
2. Processes and formats these events for blockchain storage
3. Optionally stores full event details in IPFS for space efficiency
4. Submits events to the Hyperledger Fabric blockchain
5. Provides APIs for querying the blockchain audit trail

## Setup Instructions

### Prerequisites

- Hyperledger Fabric network set up and running
- Admin identity enrolled and stored in wallet
- Node.js 16+ installed
- Required npm packages installed

### Initial Setup

1. **Set up Hyperledger Fabric Network**

```bash
cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network
./network.sh up createChannel -c neurashield-channel -ca
```

2. **Deploy Chaincode**

```bash
./network.sh deployCC -ccn neurashield -ccp /home/jub/Cursor/neurashield/backend/chaincode -ccl go -c neurashield-channel
```

3. **Enroll Admin**

```bash
cd /home/jub/Cursor/neurashield/backend
npm run enroll-admin
```

4. **Configure Environment**

Create a `.env` file in the backend directory with the following content:

```
# Blockchain Integration Configuration
CONNECTION_PROFILE_PATH=./connection-profile.json
WALLET_PATH=./wallet
CHANNEL_NAME=neurashield-channel
CHAINCODE_NAME=neurashield
BLOCKCHAIN_IDENTITY=admin
POLLING_ENABLED=true
POLLING_INTERVAL_MS=60000
IPFS_ENABLED=false
IPFS_API=http://localhost:5001/api/v0
IPFS_GATEWAY=http://localhost:8080/ipfs/
BLOCKCHAIN_WEBHOOK_URL=http://localhost:3000/api/v1/ai-detection
AI_API_URL=http://localhost:8000
```

### Starting the Service

```bash
npm install
npm start
```

## Integration Testing

To test the integration between the AI system and blockchain:

```bash
npm run test-integration
```

This will:
1. Generate sample threat events
2. Send them to the integration service
3. Verify they are correctly stored in the blockchain

## API Endpoints

### Webhook Endpoint

- **POST /api/v1/ai-detection**
  - Receives threat events from the AI system
  - Request body: Threat detection details
  - Response: 202 Accepted

### Query Endpoints

- **GET /api/v1/events**
  - Returns all events from the blockchain
  - Response: Array of events

- **GET /api/v1/events/:id**
  - Returns a specific event by ID
  - Response: Event object

## Component Architecture

The integration consists of the following components:

1. **Webhook Handler**
   - Receives real-time notifications from the AI system

2. **Polling Service**
   - Alternative method that periodically checks for new threats

3. **Event Processing**
   - Formats events for blockchain storage
   - Handles IPFS storage for full details if enabled

4. **Blockchain Submission**
   - Manages connections to the Fabric network
   - Handles transaction submission with retry logic

5. **Query Service**
   - Provides API endpoints to query the blockchain

## Error Handling

The integration includes:
- Automatic retries for failed submissions
- Queuing system for reliable event processing
- Comprehensive logging of errors and warnings
- Graceful degradation when components are unavailable

## Troubleshooting

Common issues and their solutions:

1. **Identity not found in wallet**
   - Run `npm run enroll-admin` to create the admin identity

2. **Connection refused to Fabric network**
   - Ensure the Fabric network is running
   - Check the connection profile paths

3. **Failed to submit transaction**
   - Check if the chaincode is correctly deployed
   - Verify the channel name and chaincode name

4. **IPFS errors**
   - If IPFS integration is enabled, ensure IPFS node is running
   - Consider disabling IPFS if not needed (`IPFS_ENABLED=false`)

## Security Considerations

This integration includes several security measures:
- Secure connection to the Fabric network
- Request validation and sanitization
- Separation of concerns between components
- Proper error handling to prevent information leakage 