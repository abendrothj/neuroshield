#!/bin/bash

# NeuraShield Threat Detection Daemon Runner Script
# This script runs the threat detection daemon with proper environment setup

# Set up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="$PROJECT_ROOT/models/trained/threat_detection_model.h5"
LOGS_DIR="$PROJECT_ROOT/logs"
VIRTUAL_ENV_PATH="$PROJECT_ROOT/venv"

# Make sure logs directory exists
mkdir -p "$LOGS_DIR"

# Check for virtual environment
if [ -d "$VIRTUAL_ENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VIRTUAL_ENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at $VIRTUAL_ENV_PATH"
    echo "Proceeding with system Python..."
fi

# Set environment variables
export MODEL_PATH="$MODEL_PATH"
export MONITORING_INTERVAL="5.0"
export DATA_SOURCE="api"
export DATA_API_URL="http://localhost:5000/api/v1/network-data"
export BLOCKCHAIN_API_URL="http://localhost:5000/api/v1/events"
export BLOCKCHAIN_ENABLED="true"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create a unique log file for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/threat_daemon_$TIMESTAMP.log"

echo "Starting NeuraShield Threat Detection Daemon..."
echo "Logs will be written to $LOG_FILE"

# Run the daemon
cd "$PROJECT_ROOT"
python "$PROJECT_ROOT/models/threat_detection_daemon.py" \
    --model-path "$MODEL_PATH" \
    --interval 5.0 \
    --data-source api \
    --api-url "http://localhost:5000/api/v1/network-data" 2>&1 | tee "$LOG_FILE"

# If we get here, the daemon has stopped
echo "Daemon stopped. See $LOG_FILE for details."

# Deactivate virtual environment if it was activated
if [ -d "$VIRTUAL_ENV_PATH" ]; then
    deactivate 2>/dev/null
fi 