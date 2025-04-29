# NeuraShield Blockchain Integration

This document provides an overview of the blockchain integration component of the NeuraShield project, which provides immutable logging of security events detected by the AI system.

## Overview

The blockchain integration provides:

- Immutable logging of security events detected by the AI system
- Cryptographic verification of event integrity
- Decentralized storage of critical security information
- Tamper-proof audit trails for compliance and forensics

## Components

### Chaincode (Smart Contract)

The Go-based smart contract (`/backend/chaincode/chaincode.go`) provides the business logic for:

- `InitLedger`: Initialize the blockchain with a genesis event
- `LogEvent`: Record security events with metadata
- `QueryEvent`: Retrieve specific events by ID
- `QueryAllEvents`: Retrieve all events from the ledger

The chaincode stores security events with the following structure:
```json
{
  "ID": "unique-event-id",
  "Timestamp": "ISO-8601-timestamp",
  "Type": "EventType",
  "Details": "JSON-structured-event-details",
  "IPFSHash": "IPFS-content-identifier"
}
```

### Blockchain Integration Service

The Node.js-based integration service (`/backend/src/blockchain-integration.js`) handles:

- Connecting to the Hyperledger Fabric network
- Processing security events from the AI system
- Classifying events based on threat level
- Submitting events to the blockchain with retry logic
- Optional IPFS integration for detailed event data

Key features:
- Processing queue with retry capability
- Gateway connection pooling for efficiency
- Fallback mechanisms for service unavailability

### Fabric Network Configuration

The Docker Compose configuration (`/blockchain/network/docker-compose-fabric.yml`) sets up a development Hyperledger Fabric network with:

- Orderer service
- Two peer organizations
- CLI container for administrative tasks

### REST API

The REST API (`/backend/src/server.js`) exposes endpoints for:

- `/api/v1/ai-detection`: Webhook for AI detection events
- `/api/events`: Recording and retrieving events
- `/api/health`: System health status

## Deployment

### Development Environment

To set up and run the blockchain in the development environment:

1. Set up the Hyperledger Fabric environment:
   ```bash
   cd /home/jub/Cursor/neurashield/fabric-setup
   ./bootstrap.sh
   ```

2. Start the Fabric network:
   ```bash
   cd /home/jub/Cursor/neurashield/blockchain/network
   docker-compose -f docker-compose-fabric.yml up -d
   ```

3. Deploy the chaincode:
   ```bash
   cd /home/jub/Cursor/neurashield
   ./scripts/deploy-chaincode.sh
   ```

### Production Environment

For production deployment using Kubernetes:

1. Deploy the blockchain network:
   ```bash
   cd /home/jub/Cursor/neurashield/k8s
   kubectl apply -f blockchain-deployment.yaml
   kubectl apply -f fabric-configmaps.yaml
   ```

2. Update the chaincode:
   ```bash
   cd /home/jub/Cursor/neurashield
   ./scripts/update-chaincode-k8s.sh --version 1.0
   ```

## Integration with AI Model

The AI model is integrated with the blockchain using a logging function in `/models/predict.py` that:

1. Receives threat detection results
2. Formats the events with appropriate metadata
3. Sends the events to the blockchain API
4. Implements retry logic for handling failures

The integration is configurable through environment variables:
- `BLOCKCHAIN_ENABLED`: Enable/disable blockchain logging
- `BLOCKCHAIN_API_URL`: URL of the blockchain API endpoint

## Testing

To test the blockchain integration:

1. Run the end-to-end integration test:
   ```bash
   ./test-integration.sh
   ```

2. Test the AI model with blockchain integration:
   ```bash
   cd /home/jub/Cursor/neurashield/models
   source venv/bin/activate
   export BLOCKCHAIN_ENABLED=true
   export BLOCKCHAIN_API_URL=http://localhost:3000/api/v1/events
   python test_api.py
   ```

3. Check the blockchain logs:
   ```bash
   curl http://localhost:3000/api/v1/events
   ```

## Security Considerations

1. **MSP Credentials**: Properly secure your MSP credentials
2. **TLS**: Enable TLS for all communications
3. **Access Control**: Implement proper access control for APIs
4. **Identity Management**: Secure management of blockchain identities

## Troubleshooting

If you encounter blockchain connection issues:

1. Check if the Fabric network is running:
   ```bash
   docker ps | grep hyperledger
   ```

2. Check the logs:
   ```bash
   docker logs peer0.org1.example.com
   ```

3. Verify the connection profile:
   ```bash
   cat /home/jub/Cursor/neurashield/backend/connection-profile.json
   ```

4. Check if the admin is enrolled:
   ```bash
   ls -la /home/jub/Cursor/neurashield/backend/wallet
   ```

For more details, see the comprehensive `IMPLEMENTATION_GUIDE.md` in the project root. 