# NeuraShield

NeuraShield is an advanced network threat detection system that combines neural network-based threat detection with blockchain-based audit logging. The system provides real-time threat detection, comprehensive security event logging, and an interactive dashboard for monitoring and analysis.

## Features

- **AI-Powered Threat Detection**: Utilizes multi-dataset transfer learning for accurate threat detection
- **Real-time Inference**: Processes network traffic in real-time with low latency
- **Blockchain Audit Trail**: Immutable logging of security events using Hyperledger Fabric
- **RESTful API**: Easy integration with existing security infrastructure
- **Interactive Dashboard**: Real-time visualization of threats and system status
- **Scalable Architecture**: Containerized deployment with Kubernetes support

## System Architecture

The system consists of several key components:

- **AI Engine**: Neural network-based threat detection
- **Blockchain Network**: Hyperledger Fabric for immutable event logging
- **Backend Services**: REST API and event processing
- **Frontend Dashboard**: Real-time monitoring interface
- **Monitoring System**: Performance and health metrics

## Directory Structure

```
neurashield/
├── backend/              # Backend services and API
├── blockchain/          # Blockchain implementation
├── data/               # Datasets and training data
├── docs/               # Documentation
├── fabric-setup/       # Hyperledger Fabric setup
├── frontend/           # Dashboard and UI
├── k8s/                # Kubernetes configurations
├── models/             # Trained models and model code
├── monitoring/         # Monitoring and metrics
├── output/             # Output files and logs
├── scripts/            # Utility scripts
│   ├── backup/        # Backup and recovery
│   ├── deploy/        # Deployment scripts
│   ├── setup/         # Setup and installation
│   ├── test/          # Testing scripts
│   └── utils/         # Utility functions
└── tests/             # Test files
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- Go 1.16+ (for blockchain development)
- Kubernetes cluster (for production deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neurashield.git
cd neurashield
```

2. Run the setup script:
```bash
./scripts/setup/setup.sh
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Development

1. Start the development environment:
```bash
./scripts/setup/setup.sh --dev
```

2. Run tests:
```bash
./scripts/test/test.sh
```

### Production

1. Deploy to Kubernetes:
```bash
./scripts/deploy/deploy.sh k8s
```

2. Deploy blockchain network:
```bash
./scripts/deploy/deploy.sh blockchain
```

## API Documentation

The API provides the following endpoints:

- `POST /api/v1/predict`: Submit network traffic for threat detection
- `GET /api/v1/events`: Retrieve security events
- `GET /api/v1/health`: System health check
- `GET /api/v1/metrics`: System performance metrics

## Dashboard

The dashboard provides:
- Real-time threat visualization
- Historical event analysis
- System performance metrics
- Configuration management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CICIDS2017 Dataset
- UNSW-NB15 Dataset
- Hyperledger Fabric
- TensorFlow
- React 