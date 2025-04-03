# NeuraShield

A blockchain-powered cybersecurity platform enhanced with AI threat detection.

## Project Overview

NeuraShield combines blockchain security, AI-powered threat detection, and real-time monitoring to provide a comprehensive cybersecurity solution. The system records security events on blockchain for immutability, uses advanced AI models to detect threats, and provides a user-friendly dashboard for monitoring and analysis.

## Architecture

The system consists of three main components:

1. **Frontend** - A Next.js-based web application providing dashboards and visualizations
2. **Backend** - An Express.js server that connects to the blockchain network and coordinates with the AI service
3. **AI Service** - A FastAPI service running the threat detection models and providing analysis capabilities

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.9+ (for local AI model development)

### Running the Application

#### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/neurashield.git
cd neurashield

# Start the entire stack
docker-compose up

# Access the application at http://localhost:3000
```

#### Local Development

```bash
# Backend
cd backend
npm install
npm run dev

# Frontend
cd frontend
npm install
npm run dev

# AI Service
cd ai_models
pip install -r requirements.txt
python -m ai_models.main
```

## Monitoring

The system includes Prometheus and Grafana for monitoring:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3003 (admin/admin)

## Kubernetes Deployment

For production deployment, Kubernetes configuration files are provided in the `k8s` directory:

```bash
# Apply Kubernetes configurations
kubectl apply -f k8s/
```

## Environment Variables

Each service requires specific environment variables. Example files are provided:

- `backend/.env.example`
- `frontend/.env.example`
- AI service uses environment variables in the Kubernetes deployment files

## Project Structure

```
neurashield/
├── frontend/               # Next.js web application
├── backend/                # Express.js server
├── ai_models/              # AI threat detection models
├── k8s/                    # Kubernetes deployment files
├── monitoring/             # Prometheus configuration
├── docker-compose.yml      # Docker Compose configuration
└── Dockerfile              # Multi-stage Dockerfile
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Features

- 🔒 Real-time AI threat detection
- ⛓️ Immutable event logging on Hyperledger Fabric
- 🌐 Decentralized storage using IPFS
- 📊 Real-time dashboard with WebSocket updates
- 🤖 Automated response system
- 📱 Modern, responsive UI

## Tech Stack

- **Frontend**: Next.js, React, TailwindCSS
- **Backend**: Node.js, Express
- **Blockchain**: Hyperledger Fabric
- **Storage**: IPFS
- **AI/ML**: TensorFlow/PyTorch
- **Infrastructure**: Docker, Kubernetes

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Hyperledger Fabric
- IPFS
- Next.js
- TensorFlow/PyTorch 