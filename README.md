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

For detailed setup instructions, please refer to [SETUP.md](SETUP.md).

### Prerequisites

- Docker and Docker Compose (latest version with `docker compose` command)
- Node.js 18+ (for local development)
- Python 3.9+ (for local AI model development)
- bash shell

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/neurashield.git
cd neurashield

# Make scripts executable
bash scripts/make-scripts-executable.sh

# Set up development environment
npm run setup:dev

# Start the development environment
npm run start:dev

# Access the application at http://localhost:3000
```

### Development & Production Modes

We support both development and production modes:

- **Development Mode**: `npm run setup:dev` followed by `npm run start:dev`
- **Production Mode**: `npm run setup:prod` followed by `npm run start:prod`

## Monitoring

The system includes Prometheus and Grafana for monitoring:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3003 (admin/admin in dev mode, secure credentials in production)

## Kubernetes Deployment

For production deployment, Kubernetes configuration files are provided in the `k8s` directory:

```bash
# Configure for production
npm run setup:prod

# Deploy to Kubernetes
npm run deploy:k8s
```

## Testing

The system includes comprehensive testing:

```bash
# Run all tests
npm test

# Run specific test suites
npm run test:websocket
npm run test:performance:ai-model
npm run test:security:sql-injection
npm run test:load:api

# Run system-wide tests
npm run system-test
npm run perf-test
npm run security-test
```

## Project Structure

```
neurashield/
├── frontend/               # Next.js web application
├── backend/                # Express.js server
├── ai_models/              # AI threat detection models
├── k8s/                    # Kubernetes deployment files
├── monitoring/             # Prometheus/Grafana configuration
├── scripts/                # Helper scripts
├── docker-compose.yml      # Docker Compose configuration
└── Dockerfile              # Multi-stage Dockerfile
```

## Features

- 🔒 Real-time AI threat detection
- ⛓️ Immutable event logging on Hyperledger Fabric
- 🌐 Decentralized storage using IPFS
- 📊 Real-time dashboard with WebSocket updates
- 🤖 Automated response system
- 📱 Modern, responsive UI
- 🔄 Support for both development and production environments
- 📈 Comprehensive monitoring and logging
- 🔐 Security-hardened configuration
- 💾 Database integration with PostgreSQL
- 🔄 Automated backup and recovery

## Tech Stack

- **Frontend**: Next.js 14, React, TypeScript, TailwindCSS
- **Backend**: Node.js 18.x, Express, PostgreSQL
- **Blockchain**: Hyperledger Fabric
- **Storage**: IPFS
- **AI/ML**: TensorFlow/PyTorch, FastAPI
- **Infrastructure**: Docker, Kubernetes, Prometheus, Grafana

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hyperledger Fabric
- IPFS
- Next.js
- TensorFlow/PyTorch 