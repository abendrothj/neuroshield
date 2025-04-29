# NeuraShield Scripts

This directory contains all operational and management scripts for the NeuraShield platform.

## Directory Structure

- `/blockchain/` - Scripts for blockchain network setup and chaincode deployment
  - `deploy-chaincode.sh` - Deploys chaincode to Fabric network
  - `deploy-neurashield.sh` - Deploys NeuraShield-specific chaincode
  - `deploy-simple.sh` - Deploys simple test chaincode
  - `setup-fabric-network.sh` - Sets up the Fabric network

- `/server/` - Scripts for server management
  - `run-server.sh` - Starts the NeuraShield server
  - `run-real-blockchain.sh` - Starts server with real blockchain integration
  - `reset-admin.sh` - Resets admin identity for blockchain access

- `/deployment/` - Scripts for deployment and configuration
  - `production-setup.sh` - Sets up production environment
  - `new-env-settings.sh` - Updates environment settings

- `/testing/` - Scripts for running tests
  - `test-integration.sh` - Runs integration tests
  - `test-simple.sh` - Runs simple verification tests

- Additional directories:
  - `/backup/` - Backup and recovery scripts
  - `/data/` - Data management scripts
  - `/setup/` - Initial setup scripts
  - `/utils/` - Utility scripts

## Usage

Most scripts should be run from the project root directory:

```bash
# Example: Start the server
bash scripts/server/run-server.sh

# Example: Deploy chaincode
bash scripts/blockchain/deploy-neurashield.sh
```

See the README file in each subdirectory for specific details about the scripts in that category.
