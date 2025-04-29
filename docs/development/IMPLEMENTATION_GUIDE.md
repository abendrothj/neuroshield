# NeuraShield Implementation Guide

This guide provides step-by-step instructions for implementing and running the NeuraShield system, which combines AI-powered threat detection with blockchain-based audit logging.

## Prerequisites

- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- Go 1.16+ (for blockchain development)
- Kubernetes cluster (for production deployment)

## Step 1: Set Up the Development Environment

1. Clone the repository and navigate to the project directory:

```bash
cd /home/jub/Cursor/neurashield
```

2. Set up the Hyperledger Fabric environment:

```bash
cd /home/jub/Cursor/neurashield/fabric-setup
./bootstrap.sh
```

This script downloads the necessary Hyperledger Fabric binaries and Docker images.

## Step 2: Start the Blockchain Network

1. Start the Hyperledger Fabric network:

```bash
cd /home/jub/Cursor/neurashield/blockchain/network
docker-compose -f docker-compose-fabric.yml up -d
```

2. Deploy the chaincode:

```bash
cd /home/jub/Cursor/neurashield
chmod +x scripts/deploy-chaincode.sh
./scripts/deploy-chaincode.sh
```

## Step 3: Set Up the Backend

1. Install Node.js dependencies for the backend:

```bash
cd /home/jub/Cursor/neurashield/backend
npm install
```

2. Enroll the admin user:

```bash
node enroll-admin.js
```

3. Start the backend server:

```bash
cd /home/jub/Cursor/neurashield/backend
npm start
```

This will start the backend server on port 3000.

## Step 4: Set Up the AI Model

1. Create and activate a Python virtual environment:

```bash
cd /home/jub/Cursor/neurashield/models
python -m venv venv
source venv/bin/activate
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Test the model:

```bash
python test_api.py
```

## Step 5: Integration Testing

1. Start the backend in one terminal:

```bash
cd /home/jub/Cursor/neurashield/backend
npm start
```

2. Test the AI model with blockchain integration in another terminal:

```bash
cd /home/jub/Cursor/neurashield/models
source venv/bin/activate
export BLOCKCHAIN_ENABLED=true
export BLOCKCHAIN_API_URL=http://localhost:3000/api/v1/events
python test_api.py
```

3. Check the blockchain logs:

```bash
curl http://localhost:3000/api/v1/events
```

## Step 6: Production Deployment

For production deployment using Kubernetes:

1. Deploy the blockchain network:

```bash
cd /home/jub/Cursor/neurashield/k8s
kubectl apply -f blockchain-deployment.yaml
kubectl apply -f fabric-configmaps.yaml
```

2. Update the chaincode:

```bash
cd /home/jub/Cursor/neurashield
./scripts/update-chaincode-k8s.sh --version 1.0
```

3. Deploy the backend:

```bash
cd /home/jub/Cursor/neurashield/k8s
kubectl apply -f backend-deployment.yaml
```

4. Deploy the AI model:

```bash
cd /home/jub/Cursor/neurashield/k8s
kubectl apply -f ai-model-deployment.yaml
```

## Troubleshooting

### Blockchain Connection Issues

If you encounter blockchain connection issues:

1. Check if the Fabric network is running:

```bash
docker ps | grep hyperledger
```

2. Check the logs:

```bash
docker logs peer0.org1.example.com
```

3. Verify the connection profile:

```bash
cat /home/jub/Cursor/neurashield/backend/connection-profile.json
```

### Identity Management Issues

If you have identity management issues:

1. Check if the admin is enrolled:

```bash
ls -la /home/jub/Cursor/neurashield/backend/wallet
```

2. Re-enroll the admin:

```bash
cd /home/jub/Cursor/neurashield/backend
node enroll-admin.js
```

### AI Model Issues

If the AI model is not working correctly:

1. Check the model logs:

```bash
cat /home/jub/Cursor/neurashield/models/logs/threat_detection.log
```

2. Verify the model file exists:

```bash
ls -la /home/jub/Cursor/neurashield/models/*.keras
```

## Security Considerations

1. **Production Environment**:
   - Always use TLS for secure communication
   - Store Fabric certificates securely
   - Use proper access control for APIs

2. **MSP Management**:
   - Securely manage MSP credentials
   - Implement proper identity management

3. **API Security**:
   - Implement authentication for all API endpoints
   - Use HTTPS for all communications

## Next Steps

1. **Frontend Development**:
   - Develop the dashboard for visualizing security events
   - Implement the blockchain explorer component

2. **Advanced Features**:
   - Set up monitoring for blockchain network
   - Implement federated learning capabilities
   - Add cross-organization threat intelligence sharing

3. **Additional AI Models**:
   - Train models on additional datasets
   - Implement advanced anomaly detection algorithms 