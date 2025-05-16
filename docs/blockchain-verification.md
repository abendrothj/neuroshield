# NeuraShield Blockchain Verification System

The NeuraShield Blockchain Verification System provides cryptographic proof of security events, ensuring data integrity and immutability using blockchain technology.

## Overview

The blockchain verification system consists of several components:

1. **Blockchain Adapter**: Interface between NeuraShield and the Hyperledger Fabric blockchain
2. **Event Hashing**: Cryptographic hashing mechanism to generate unique fingerprints of events
3. **Verification Engine**: System to validate event integrity and blockchain transactions
4. **Verification Certificates**: Tamper-proof records confirming event authenticity

## How Verification Works

When a security event is detected by NeuraShield, the following steps occur:

1. The event data is captured with detailed metadata
2. A cryptographic hash of the event is generated using SHA-256
3. The event and its hash are recorded on the blockchain
4. A transaction receipt is created with the blockchain transaction ID
5. The event can be verified at any time to prove its authenticity

## Verification Process

The verification process checks two critical aspects:

### 1. Data Integrity

To verify data integrity, the system:
- Recalculates the cryptographic hash of the event data
- Compares it with the hash stored on the blockchain
- Confirms the data has not been modified since recording

### 2. Blockchain Proof

To verify blockchain proof, the system:
- Retrieves the transaction details from the blockchain
- Validates the transaction exists and is valid
- Confirms the transaction contains the correct event data
- Verifies the block containing the transaction is part of the chain

## Verification API

### Endpoint

```
GET /api/blockchain/verify/<event_id>
```

### Response

The verification endpoint returns a verification certificate with the following information:

```json
{
  "eventId": "event-12345",
  "type": "SECURITY_EVENT",
  "timestamp": "2023-06-15T10:30:15.123Z",
  "blockchainTimestamp": "2023-06-15T10:30:45.789Z",
  "blockNumber": "12345",
  "transactionId": "abcdef1234567890abcdef1234567890",
  "dataHash": "8a1bc7a5d0e96c273d974b3b713f11067193ebca4efe7266e249d7774ebcc67d",
  "verificationTimestamp": "2023-06-15T15:45:30.456Z",
  "status": "VERIFIED",
  "verificationDetails": {
    "dataIntegrity": {
      "isValid": true,
      "calculatedHash": "8a1bc7a5d0e96c273d974b3b713f11067193ebca4efe7266e249d7774ebcc67d",
      "storedHash": "8a1bc7a5d0e96c273d974b3b713f11067193ebca4efe7266e249d7774ebcc67d"
    },
    "transactionValidity": {
      "isValid": true,
      "txId": "abcdef1234567890abcdef1234567890",
      "blockNumber": "12345",
      "timestamp": "2023-06-15T10:30:45.789Z",
      "validationCode": 0
    }
  }
}
```

## Tamper Detection

The blockchain verification system provides robust tamper detection capabilities:

1. **Data Tampering**: Any modification to the event data will result in a different hash, which will not match the stored hash, indicating tampering.

2. **Database Tampering**: Since the hash is stored on the blockchain, manipulating the database will not affect the verification process.

3. **Unauthorized Events**: Events not properly recorded on the blockchain will fail verification.

## Frontend Verification Component

NeuraShield includes a user-friendly verification component that allows users to:

- Verify any security event using its ID
- View detailed verification information
- See visual confirmation of verification status
- Access the full verification certificate

## Technical Implementation

The blockchain verification system is built using:

- **Hyperledger Fabric**: Enterprise-grade permissioned blockchain
- **SHA-256**: Cryptographic hashing algorithm
- **Smart Contracts**: Custom chaincode for event logging and verification
- **Integration Layer**: Node.js adapter to connect the application to the blockchain

## Development and Testing

To test the verification system locally:

1. Run the blockchain network:
   ```
   cd backend
   ./install-fabric.sh
   ```

2. Test event verification:
   ```
   cd test
   node blockchain_verification_test.js
   ```

3. View verification component in the UI:
   ```
   cd frontend
   npm run dev
   ```
   Then navigate to `/blockchain/verify` in the browser.

## Security Considerations

- Keep blockchain access credentials secure
- Regularly update blockchain nodes and libraries
- Monitor for unusual verification patterns that might indicate attacks
- Implement proper access controls for the verification API 