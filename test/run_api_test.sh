#!/bin/bash

# Setup environment variables
export MODEL_PATH="/home/jub/Cursor/neurashield/models/trained/threat_detection_model.h5"
export MONITORING_INTERVAL="2.0"
export DATA_SOURCE="api"
export DATA_API_URL="http://localhost:5000/api/v1/network-data"
export BLOCKCHAIN_API_URL="http://localhost:5000/api/v1/events"
export BLOCKCHAIN_ENABLED="true"

# Run the daemon with API data source
echo "Running threat detection daemon with API data source..."
python3 /home/jub/Cursor/neurashield/models/threat_detection_daemon.py \
    --model-path "$MODEL_PATH" \
    --interval 2.0 \
    --data-source api \
    --api-url "$DATA_API_URL" 