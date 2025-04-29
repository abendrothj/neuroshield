# Blockchain Scripts

This directory contains scripts related to the Hyperledger Fabric blockchain setup and management.

## Scripts

- `deploy-chaincode.sh` - Deploys chaincode to an existing Fabric network
- `deploy-neurashield.sh` - Deploys the NeuraShield-specific chaincode to the Fabric network
- `deploy-simple.sh` - Deploys simple test chaincode for verification purposes
- `setup-fabric-network.sh` - Sets up the Fabric network environment

## Usage

Most scripts should be run from the project root directory:

```bash
# Example: Deploy chaincode
bash scripts/blockchain/deploy-chaincode.sh

# Example: Setup the network
bash scripts/blockchain/setup-fabric-network.sh
```

For more details on the blockchain integration, refer to the architecture documentation. 