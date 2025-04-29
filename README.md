# NeuraShield

NeuraShield is a security platform that combines blockchain technology with AI to create an immutable security event management system.

## Project Structure

- `/backend` - Node.js backend server with blockchain integration
- `/blockchain` - Blockchain configuration and network setup
- `/configs` - Configuration files and environment settings
- `/docs` - Project documentation
  - `/architecture` - System architecture and design docs
  - `/development` - Development guides and specifications
  - `/user_guides` - User guides and deployment instructions
- `/frontend` - React.js frontend application
- `/k8s` - Kubernetes deployment configurations
- `/models` - AI/ML models for security event analysis
- `/monitoring` - System monitoring configuration
- `/scripts` - Operational and management scripts
  - `/blockchain` - Blockchain management scripts
  - `/deployment` - Deployment and setup scripts
  - `/server` - Server management scripts
  - `/testing` - Test scripts

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js 16+
- Python 3.10+ (for AI models)
- Hyperledger Fabric 2.4+

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-org/neurashield.git
   cd neurashield
   ```

2. Set up the Fabric network:
   ```
   bash scripts/blockchain/setup-fabric-network.sh
   ```

3. Deploy the chaincode:
   ```
   bash scripts/blockchain/deploy-neurashield.sh
   ```

4. Start the server:
   ```
   bash scripts/server/run-server.sh
   ```

5. Access the application at http://localhost:3000

## Documentation

For more detailed information, refer to the documentation in the `/docs` directory:

- Architecture & Design: `/docs/architecture/`
- Development Guide: `/docs/development/`
- Deployment Instructions: `/docs/user_guides/`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 