# NeuraShield

A cutting-edge cybersecurity platform that combines AI-powered threat detection with blockchain-based immutable logging.

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

## Getting Started

### Prerequisites

- Node.js (v18 or later)
- Python (v3.8 or later)
- Docker and Docker Compose
- Go (v1.20 or later)
- IPFS

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neurashield.git
cd neurashield
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
cd backend
npm install
```

4. Set up environment variables:
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

5. Start the development servers:
```bash
# Terminal 1 - Frontend
npm run dev

# Terminal 2 - Backend
cd backend
npm run dev
```

## Project Structure

```
neurashield/
├── app/                    # Next.js frontend pages
├── components/            # React components
├── backend/              # Node.js backend
├── lib/                  # Shared utilities
├── public/              # Static assets
└── styles/              # Global styles
```

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