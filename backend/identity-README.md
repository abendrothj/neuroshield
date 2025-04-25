# NeuraShield Identity Management

This module provides a comprehensive identity management system for the NeuraShield blockchain integration. It handles the enrollment, registration, and verification of identities for interacting with the Hyperledger Fabric blockchain network.

## Features

- **Admin Enrollment**: Multiple methods to enroll the admin user (MSP or CA)
- **User Registration**: Register and enroll regular users
- **Service Identities**: Create special identities for automated services
- **Identity Verification**: Verify identities can connect to the network
- **Identity Management**: List and delete identities from the wallet
- **CLI Tool**: Command-line interface for easy management

## Setup

Before using the identity management system, ensure you have:

1. A running Hyperledger Fabric network
2. Properly configured connection profile
3. Required environment variables set (or use defaults in the code)

## Usage

### Command Line Interface

The identity management system includes a CLI tool for easy management:

```bash
# Initialize all required identities (admin and blockchain-service)
npm run identity init

# Enroll admin user only
npm run identity enroll-admin

# Register a new user
npm run identity register john.doe

# Create a service identity
npm run identity service api-service

# List all identities in the wallet
npm run identity list

# Verify an identity works with the network
npm run identity verify blockchain-service

# Delete an identity
npm run identity delete john.doe
```

### Programmatic Usage

You can also use the identity management system programmatically in your code:

```javascript
const identityManager = require('./identity-manager');

// Initialize all required identities
await identityManager.initializeIdentities();

// Register a new user
await identityManager.registerUser('john.doe');

// Get a gateway connection with a specific identity
const gateway = await identityManager.getGatewayConnection('blockchain-service');

// Use the gateway for blockchain operations
const network = await gateway.getNetwork('neurashield-channel');
const contract = network.getContract('neurashield');

// Always disconnect when done
gateway.disconnect();
```

## Environment Variables

The identity management system uses the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| CONNECTION_PROFILE_PATH | ./connection-profile.json | Path to connection profile |
| WALLET_PATH | ./wallet | Path to the wallet directory |
| MSP_ID | Org1MSP | MSP ID for the organization |
| CA_URL | https://localhost:7054 | URL for the certificate authority |
| ADMIN_IDENTITY | admin | Label for the admin identity |
| ADMIN_PASSWORD | adminpw | Password for the admin user (CA enrollment) |
| ADMIN_MSP_PATH | /path/to/fabric-samples/... | Path to MSP materials |

## Identity Types

### Admin Identity

The admin identity has full administrative privileges on the blockchain network. It's required for registering other users and can be enrolled in two ways:

1. **MSP Enrollment**: Uses existing MSP materials from the network setup
2. **CA Enrollment**: Enrolls directly with the Certificate Authority

### Service Identities

Service identities are specialized identities for automated processes, such as:

- The blockchain integration service
- API services
- Monitoring services

These identities have specific attributes that limit their permissions to what they need to function.

### User Identities

Regular user identities represent human users and are registered and enrolled through the admin identity. These identities can have different roles and affiliations.

## Security Considerations

1. **Wallet Security**: The wallet contains sensitive credentials. Ensure the directory is properly secured.
2. **Admin Password**: If using CA enrollment, keep the admin password secure.
3. **Identity Rotation**: Consider rotating service identities periodically.
4. **Access Control**: Use appropriate roles and attributes to limit permissions.

## Troubleshooting

### Common Issues

1. **MSP Path Not Found**
   - Check that the MSP materials are in the expected location
   - Verify file permissions

2. **CA Connection Failure**
   - Ensure the CA is running and accessible
   - Check TLS certificates if using TLS

3. **Identity Already Exists**
   - Use `identity delete` to remove existing identity before recreating
   - Or use the existing identity if it's valid

4. **Network Connection Failure**
   - Verify the connection profile is correct
   - Check that the network is running and accessible 