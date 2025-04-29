# NeuraShield Project Context

## Overview
NeuraShield is a blockchain-based system built on Hyperledger Fabric for secure AI threat detection and logging. The system uses a Fabric network to securely record and verify AI-generated threat detections.

## Current Setup Status

### Network Configuration
- Fabric network is running with the `neurashield-channel` channel
- Certificate Authority (CA) is active
- TLS is enabled with hostname verification disabled to prevent certificate errors

### Identity Configuration
- Admin identity certificates copied from Fabric to the application wallet directory
- Environment configured to use the admin identity for blockchain transactions
- Hostname verification disabled in connection profile to fix TLS issues

### Blockchain Integration
- Mock blockchain implementation has been removed (deleted files: `backend/mock-blockchain.js` and `run-mock-blockchain.sh`)
- System configured to use real Hyperledger Fabric network
- Simple chaincode ready for deployment

### Connection Details
- Using Organization 1 (Org1MSP) for blockchain operations
- Channel name: `neurashield-channel`
- Chaincode name: `simple`
- Contract name: `simple`

### Environment Configuration
- Production mode enabled
- TLS enabled with hostname verification disabled
- IPFS configured for localhost:5001
- Mock implementations explicitly disabled

## Next Steps
1. Complete chaincode deployment to the Fabric network
2. Verify blockchain connectivity from the application
3. Test the entire workflow with AI service integration
4. Monitor for any TLS or certificate-related issues

## Testing
Sample API call for testing AI integration:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prediction": {"timestamp": "2023-08-21T14:30:00.000Z", "threat_type": "malware", "confidence": 0.85, "affected_system": "web-server-01", "source_ip": "192.168.1.100"}}' \
  http://localhost:3000/api/v1/ai-detection
```

## Useful Commands
- Start server: `/home/jub/Cursor/neurashield/run-server.sh`
- Fix TLS issues: `/home/jub/Cursor/neurashield/fix-tls.sh`
- Deploy chaincode: `/home/jub/Cursor/neurashield/deploy-simple.sh`

## Project Overview
NeuraShield is a comprehensive cybersecurity platform that combines advanced neural network-based threat detection with immutable blockchain logging for secure audit trails. The system uses AI to detect anomalous network traffic patterns and security threats in real-time, while leveraging Hyperledger Fabric blockchain technology to create tamper-proof records of security events.

## System Architecture

### Core Components
1. **AI-Powered Threat Detection Engine**
   - Utilizes multi-dataset learning models
   - Implements transfer learning for rapid adaptation to new threats
   - Processes network traffic in real-time
   - Employs feature engineering for optimized detection

2. **Blockchain Audit Trail**
   - Hyperledger Fabric implementation
   - Immutable ledger for security events
   - Smart contract (chaincode) for event logging
   - IPFS integration for storing full logs

3. **Frontend Dashboard**
   - Real-time visualization of threats
   - Security event monitoring
   - Interactive analysis tools
   - Administrative controls

4. **Backend API Services**
   - RESTful API for system interaction
   - WebSocket for real-time updates
   - Integration with third-party security tools
   - Authentication and authorization controls

## Directory Structure
```
neurashield/
├── ai_models/             # AI model implementations
├── backend/               # Backend API and services
│   ├── chaincode/         # Hyperledger Fabric chaincode
│   ├── src/               # Backend source code
│   ├── wallet/            # Blockchain identity wallet
│   ├── test-blockchain.js # Blockchain test script
│   └── connection-profile.json # Fabric network connection profile
├── blockchain/            # Blockchain implementation
│   └── network/           # Network configuration
├── data/                  # Training and test datasets
├── docs/                  # Documentation
├── fabric-samples/        # Hyperledger Fabric examples
├── fabric-setup/          # Fabric network setup scripts
├── frontend/              # User interface
├── k8s/                   # Kubernetes deployment files
├── models/                # Trained model storage
├── monitoring/            # System monitoring tools
├── reports/               # Analytics reports
├── scripts/               # Utility scripts
└── tests/                 # Test suites
```

## Technology Stack

### AI/ML Components
- **TensorFlow/Keras**: Deep learning framework for model development
- **Scikit-learn**: For preprocessing and feature engineering
- **PyTorch**: Alternative deep learning implementation for specific models
- **CUDA**: GPU acceleration for model training
- **NumPy/Pandas**: Data manipulation and preprocessing

### Blockchain Components
- **Hyperledger Fabric**: Primary blockchain framework
- **Golang**: Chaincode implementation language
- **Node.js**: SDK for blockchain interaction
- **IPFS**: Distributed storage for full log data

### Backend Components
- **Node.js**: Primary backend runtime
- **Express**: Web framework for API
- **Socket.io**: WebSocket implementation
- **MongoDB**: Primary database for application data
- **Redis**: Cache and message broker

### Frontend Components
- **React**: UI framework
- **D3.js**: Data visualization
- **Material-UI**: Component library
- **Redux**: State management

### Infrastructure
- **Docker**: Container runtime
- **Kubernetes**: Container orchestration
- **Prometheus/Grafana**: Monitoring and observability
- **NGINX**: Web server and reverse proxy
- **Let's Encrypt**: TLS certificate management

## Detailed Component Descriptions

### 1. AI Threat Detection System

#### Model Architecture
The system employs a multi-layer neural network with specialized components:
- **Feature Extraction Layers**: Convolutional and LSTM layers to identify patterns in network traffic
- **Transfer Learning Mechanism**: Pre-trained models adapted to new threats
- **Anomaly Detection**: Autoencoders to identify deviations from normal patterns
- **Classification Layers**: For specific threat categorization

#### Datasets
The system is trained on multiple security datasets:
- CSE-CIC-IDS2018
- UNSW-NB15
- CIC-DDoS19
- Internal proprietary datasets

#### Threat Detection Pipeline
1. Network traffic data ingestion
2. Feature extraction and normalization
3. Anomaly detection
4. Threat classification
5. Severity assessment
6. Alert generation
7. Blockchain logging

### 2. Blockchain Implementation

#### Hyperledger Fabric Network
- **Organizations**: Two-organization network (expandable)
- **Peers**: Multiple peer nodes per organization
- **Orderers**: Raft consensus mechanism
- **Channel**: "neurashield-channel" for security events
- **CA**: Certificate Authorities for each organization

#### Chaincode (Smart Contract)
- Written in Go
- Core functions:
  - `InitLedger`: Initialize the ledger with default values
  - `LogEvent`: Record security events with metadata
  - `QueryEvent`: Retrieve specific event details
  - `QueryAllEvents`: Retrieve all events from the ledger

#### Event Structure
```json
{
  "id": "unique-event-id",
  "timestamp": "ISO-8601-timestamp",
  "type": "EventType",
  "details": "JSON-structured-event-details",
  "ipfsHash": "IPFS-CID-for-full-logs"
}
```

#### Integration with Backend
The backend connects to the Fabric network using:
- Fabric SDK for Node.js
- Identity wallet management
- TLS secure connections
- Transaction submission and query capabilities

### 3. Backend API Services

#### Core APIs
- `/auth`: Authentication and authorization
- `/ai`: AI model interaction and predictions
- `/blockchain`: Blockchain record management
- `/dashboard`: Dashboard data aggregation
- `/admin`: System administration
- `/events`: Event management

#### WebSocket Services
- Real-time threat notifications
- Live traffic analysis
- System health monitoring
- User session management

#### Security Features
- JWT-based authentication
- Role-based access control
- Input validation and sanitization
- Rate limiting
- Audit logging

### 4. Frontend Dashboard

#### Main Views
- **Overview**: System status and key metrics
- **Threats**: Real-time and historical threat visualization
- **Network**: Network traffic analysis
- **Blockchain**: Blockchain record explorer
- **Reports**: Customizable reporting
- **Settings**: System configuration

#### Visualization Components
- Network traffic heat maps
- Threat origin geolocation
- Time-series analysis
- Entity relationship graphs
- Severity distribution charts

## Deployment Architecture

### Production Environment
- **Kubernetes-based** orchestration
- **Multi-zone** deployment for high availability
- **Microservices architecture** for scalability
- **CI/CD pipeline** for automated deployment
- **Blue/Green deployment** strategy for updates

### Development Environment
- **Docker Compose** for local development
- **Minikube** for Kubernetes development
- **Jest/Mocha** for automated testing
- **ESLint/Prettier** for code quality

## Testing Methodology

### Unit Testing
- Individual component functionality
- Model performance metrics
- Chaincode function validation

### Integration Testing
- API endpoint integration
- Blockchain-API interaction
- Frontend-Backend communication

### System Testing
- End-to-end workflows
- Performance under load
- Security vulnerability assessment

### User Acceptance Testing
- Threat detection accuracy
- False positive/negative rates
- Dashboard usability
- Response time metrics

## Blockchain Testing Procedures

1. **Network Setup**
   ```bash
   cd fabric-setup/fabric-samples/test-network
   ./network.sh up createChannel -c neurashield-channel -ca
   ```

2. **Chaincode Deployment**
   ```bash
   ./network.sh deployCC -ccn neurashield -ccp /path/to/neurashield/backend/chaincode -ccl go -c neurashield-channel
   ```

3. **Event Logging Test**
   ```bash
   docker run --rm -it --network=fabric_test --name=cli [environment variables] \
   hyperledger/fabric-tools:latest bash -c "peer chaincode invoke [options] \
   -c '{\"function\":\"LogEvent\",\"Args\":[\"event-id\", \"timestamp\", \"type\", \"details\", \"ipfsHash\"]}'"
   ```

4. **Query Testing**
   ```bash
   docker run --rm -it --network=fabric_test --name=cli [environment variables] \
   hyperledger/fabric-tools:latest bash -c "peer chaincode query [options] \
   -c '{\"function\":\"QueryEvent\",\"Args\":[\"event-id\"]}'"
   ```

## Security Considerations

### Data Protection
- Encryption at rest and in transit
- Secure key management
- Personal data anonymization
- GDPR compliance mechanisms

### Network Security
- TLS for all communications
- mTLS for service-to-service communication
- Network segmentation
- WAF protection

### Blockchain Security
- Private permissioned network
- Certificate-based authentication
- Access control lists
- Channel-based isolation

### AI Model Security
- Adversarial training
- Model encryption
- Inference protection
- Data poisoning countermeasures

## Future Roadmap

### Near-term Enhancements
- Advanced federated learning integration
- Enhanced blockchain consensus mechanisms
- Expanded threat intelligence feeds
- Automated remediation workflows

### Long-term Vision
- Multi-cloud deployment support
- Cross-organization threat intelligence sharing
- Quantum-resistant cryptography
- Autonomous threat hunting capabilities

## Conclusion
NeuraShield represents a next-generation approach to cybersecurity by combining the pattern recognition capabilities of neural networks with the immutable audit trails provided by blockchain technology. This hybrid approach ensures both proactive threat detection and tamper-proof forensic capabilities, creating a robust security posture for modern enterprises.

The system's modular architecture allows for continuous improvement and adaptation to evolving threats, while the blockchain component ensures compliance with regulatory requirements for security event logging and non-repudiation. 