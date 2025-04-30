# NeuraShield Threat Detection Daemon

The NeuraShield Threat Detection Daemon is a continuous monitoring service that analyzes network traffic for potential threats and logs detected threats to the blockchain for immutable record-keeping.

## Features

- **Continuous Monitoring**: Runs as a background service that constantly analyzes incoming network traffic
- **Threat Detection**: Uses a trained machine learning model to detect anomalies and potential security threats
- **Blockchain Integration**: Logs detected threats to the blockchain for immutable auditing
- **Multiple Data Sources**: Can read from an API endpoint or data files
- **Configurable**: Easily adjustable settings via environment variables or command-line arguments
- **Robust Logging**: Comprehensive logging of all activities and detected threats

## Architecture

The daemon follows a modular architecture:

1. **Data Collection**: Fetches data from configured sources
2. **Preprocessing**: Normalizes and prepares data for the model
3. **Prediction**: Uses the trained model to detect threats
4. **Blockchain Logging**: Sends threat data to the blockchain
5. **Monitoring**: Maintains continuous operation and status reporting

## Installation

### Prerequisites

- Python 3.8 or later
- TensorFlow 2.x
- Required Python packages (see `requirements.txt`)
- Access to the NeuraShield API server
- Access to the NeuraShield blockchain adapter

### Setup

1. Ensure the Python virtual environment is set up:

```bash
cd /home/jub/Cursor/neurashield
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Train or download a threat detection model and place it in the `models/trained/` directory.

3. Make the run script executable:

```bash
chmod +x /home/jub/Cursor/neurashield/scripts/run_threat_detection_daemon.sh
```

## Running the Daemon

### Manual Execution

To run the daemon manually:

```bash
cd /home/jub/Cursor/neurashield
./scripts/run_threat_detection_daemon.sh
```

### Command Line Options

The daemon accepts the following command-line arguments:

- `--model-path`: Path to the trained model (default: from env var MODEL_PATH)
- `--interval`: Monitoring interval in seconds (default: 5.0)
- `--data-source`: Source of network data ('api' or 'file') (default: 'api')
- `--api-url`: URL for the network data API (default: http://localhost:5000/api/v1/network-data)
- `--input-file`: Path to input data file (default: data/network_traffic.csv)

### Running as a SystemD Service

To install and run as a systemd service:

1. Copy the service file to the systemd directory:

```bash
sudo cp /home/jub/Cursor/neurashield/scripts/systemd/neurashield-threat-detection.service /etc/systemd/system/
```

2. Reload systemd and enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable neurashield-threat-detection
sudo systemctl start neurashield-threat-detection
```

3. Check the status:

```bash
sudo systemctl status neurashield-threat-detection
```

## Integration with NeuraShield

The threat detection daemon integrates with other NeuraShield components:

1. **Data Collection**: Receives network traffic data from the monitoring API
2. **Threat Detection**: Uses trained models from the AI module
3. **Blockchain Integration**: Sends threat data to the blockchain adapter

## Environment Variables

The daemon can be configured using the following environment variables:

- `MODEL_PATH`: Path to the trained model
- `MONITORING_INTERVAL`: Time between checks in seconds
- `DATA_SOURCE`: Source of network data ('api' or 'file')
- `DATA_API_URL`: URL for the network data API
- `INPUT_FILE`: Path to input data file when using file source
- `BLOCKCHAIN_API_URL`: URL for the blockchain API
- `BLOCKCHAIN_ENABLED`: Whether to enable blockchain logging ('true' or 'false')

## Logging

The daemon logs to the following locations:

- Main application log: `/home/jub/Cursor/neurashield/logs/threat_detection_daemon.log`
- Service log (when run as systemd service): `/home/jub/Cursor/neurashield/logs/threat_detection_service.log`
- Run-specific logs: `/home/jub/Cursor/neurashield/logs/threat_daemon_[TIMESTAMP].log`

## Troubleshooting

Common issues and solutions:

1. **Model loading failure**: Ensure the model path is correct and the model is compatible with TensorFlow 2.x
2. **API connection issues**: Check network connectivity and API endpoint availability
3. **Blockchain logging failures**: Verify the blockchain adapter is running and accessible

## Future Enhancements

- Real-time alerting via email or messaging platforms
- Integration with SIEM systems
- Support for multiple models and detection techniques
- Advanced data preprocessing for improved accuracy
- Performance optimizations for large-scale deployments 