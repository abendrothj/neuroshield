#!/bin/bash

# Set environment variables
export PORT=3000
export NODE_ENV=development
export LOG_LEVEL=info
export SKIP_BLOCKCHAIN=false
export CHANNEL_NAME=neurashield-channel
export CONTRACT_NAME=neurashield
export IPFS_URL=http://localhost:5001
export IPFS_ENABLED=false
export AI_SERVICE_URL=http://localhost:8000

# Run the server
npm run dev 