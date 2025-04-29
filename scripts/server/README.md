# Server Scripts

This directory contains scripts related to the NeuraShield server management and operation.

## Scripts

- `run-server.sh` - Starts the NeuraShield server
- `run-real-blockchain.sh` - Starts the server with real blockchain integration
- `reset-admin.sh` - Resets the admin identity in the wallet for blockchain access

## Usage

Most scripts should be run from the project root directory:

```bash
# Example: Start the server
bash scripts/server/run-server.sh

# Example: Reset admin credentials when having blockchain access issues
bash scripts/server/reset-admin.sh
```

## Notes

- Make sure the Fabric network is running before starting the server with blockchain integration
- The `reset-admin.sh` script should be used when experiencing credential issues with the blockchain 