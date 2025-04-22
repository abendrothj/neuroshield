# NeuraShield: Advanced Network Threat Detection with Blockchain Audit Trail

NeuraShield is a comprehensive cybersecurity platform that combines advanced neural network-based threat detection with immutable blockchain logging for secure audit trails. By leveraging knowledge from multiple cybersecurity datasets and Hyperledger Fabric blockchain technology, NeuraShield provides exceptional detection capabilities with tamper-proof forensic evidence.

## Features

- **Multi-Dataset Transfer Learning**: Progressively builds knowledge from multiple datasets for superior threat detection
- **Real-time Inference**: Analyze network traffic in real-time with high accuracy (96%)
- **Blockchain Audit Trail**: Immutable logging of security events using Hyperledger Fabric
- **IPFS Integration**: Distributed storage for full log data
- **RESTful API**: Simple integration with existing security infrastructure
- **Dashboard**: Interactive visualizations for monitoring threats and blockchain records
- **Docker & Kubernetes Support**: Enterprise-grade deployment options

## Architecture

NeuraShield consists of several integrated components:

1. **AI-Powered Threat Detection Engine**:
   - Single-dataset training: `train_simple.py`, `train_advanced.py`
   - Multi-dataset learning: `multi_dataset_learning.py`
   - Chained transfer learning: `chain_transfer.py`
   - Feature engineering and optimization

2. **Blockchain Audit Trail**:
   - Hyperledger Fabric implementation in `blockchain/` and `fabric-setup/`
   - Smart contract (chaincode) in `backend/chaincode/`
   - Identity management in `backend/wallet/`
   - IPFS integration for storing full logs

3. **Backend Services**:
   - Core model serving: `inference.py`
   - API service: `api.py`
   - Traffic processor: `traffic_processor.py`
   - Blockchain connector: `backend/test-blockchain.js`

4. **Frontend Dashboard**:
   - Dashboard: `dashboard.py`
   - Real-time threat visualizations
   - Blockchain record explorer
   - Security event monitoring

## Performance

The current model achieves:
- 96% overall accuracy
- 0.997 AUC score (exceptional discrimination ability)
- 0.919 Matthews Correlation Coefficient
- Balanced detection (95% F1-score for benign traffic, 97% for attacks)
- Immutable and verifiable audit trail for all security events

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- Node.js 14+
- Docker and Docker Compose
- Kubernetes (for production deployment)
- Hyperledger Fabric 2.2+

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/neurashield.git
   cd neurashield
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   cd backend && npm install
   ```

3. **Set up the Fabric network**:
   ```bash
   cd fabric-setup
   ./bootstrap.sh
   cd fabric-samples/test-network
   ./network.sh up createChannel -c neurashield-channel -ca
   ./network.sh deployCC -ccn neurashield -ccp /path/to/neurashield/backend/chaincode -ccl go -c neurashield-channel
   ```

4. **Download pre-trained models** (optional):
   ```bash
   mkdir -p models/multi_dataset/chained_transfer_improved
   # Download pre-trained models here
   ```

## Usage

### Training Models

1. **Single dataset training**:
   ```bash
   python train_simple.py --dataset-path /path/to/dataset --model-type residual
   ```

2. **Multi-dataset learning**:
   ```bash
   python multi_dataset_learning.py --unsw-path /path/to/UNSW_NB15 --cic-ddos19-path /path/to/CIC-DDoS19
   ```

3. **Chained transfer learning**:
   ```bash
   python chain_transfer.py --base-model-path /path/to/first/model --target-dataset-path /path/to/CSE-CIC-IDS2018
   ```

### Running the System

1. **Start all services using Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **Or start individual services**:
   ```bash
   # Start the AI service
   python api.py
   
   # Start the dashboard
   streamlit run dashboard.py
   
   # Process network traffic
   python traffic_processor.py --interface eth0
   
   # Start the blockchain connector
   cd backend && node test-blockchain.js
   ```

### Blockchain Interaction

1. **Enroll admin user**:
   ```bash
   cd backend
   node enroll-admin.js
   ```

2. **Log a security event**:
   ```bash
   docker run --rm -it --network=fabric_test --name=cli -e GOPATH=/opt/gopath [...environment variables...] \
   hyperledger/fabric-tools:latest bash -c "peer chaincode invoke [...options...] \
   -c '{\"function\":\"LogEvent\",\"Args\":[\"event-id\", \"timestamp\", \"type\", \"details\", \"ipfsHash\"]}'"
   ```

3. **Query events**:
   ```bash
   docker run --rm -it --network=fabric_test --name=cli -e GOPATH=/opt/gopath [...environment variables...] \
   hyperledger/fabric-tools:latest bash -c "peer chaincode query [...options...] \
   -c '{\"function\":\"QueryAllEvents\",\"Args\":[]}'"
   ```

## API Documentation

The API is available at `http://localhost:8000/docs` when running, with these key endpoints:

- `GET /health`: API health check
- `POST /predict`: Make a single prediction
- `POST /predict/batch`: Make batch predictions
- `POST /explain`: Get feature contribution explanations
- `POST /blockchain/log`: Log a security event to the blockchain
- `GET /blockchain/events`: Query blockchain events

## Dashboard

The dashboard is available at `http://localhost:8501` and provides:

- Real-time threat detection visualization
- Attack trend analysis
- Feature importance insights
- Blockchain record explorer
- Audit trail verification
- Alert management

## Deployment

### Development Environment
- Docker Compose for local setup
- Minikube for Kubernetes development

### Production Environment
- Kubernetes for orchestration
- Multi-zone deployment for high availability
- Microservices architecture
- CI/CD pipeline support

## Security Considerations

- Data encryption at rest and in transit
- Private permissioned blockchain network
- Certificate-based authentication for blockchain
- Role-based access control for API
- Model security against adversarial attacks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UNSW-NB15, CIC-DDoS19, and CSE-CIC-IDS2018 datasets
- Hyperledger Fabric community
- TensorFlow team for their ML framework
- IPFS for distributed storage 