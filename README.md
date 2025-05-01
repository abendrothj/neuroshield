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

# NeuraShield Frontend

A modern, React-based frontend for the NeuraShield cybersecurity platform, featuring blockchain integration, AI detection visualization, and IPFS storage.

## Features

- **Authentication System**: Secure user authentication with JWT token management
- **Dashboard Analytics**: Real-time security event monitoring and metrics
- **Blockchain Integration**: Verification and display of blockchain-secured security events
- **AI Detection Visualization**: Interactive visualizations of AI-detected security threats
- **IPFS Evidence Storage**: Distributed, immutable storage of security evidence
- **User Management**: Role-based access control and user administration
- **Responsive Design**: Modern, responsive UI built with TailwindCSS
- **Performance Monitoring**: Built-in performance tracking and optimization tools

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **Testing**: Jest with React Testing Library
- **State Management**: React Context API
- **API Integration**: Fetch API with type-safe services
- **Authentication**: JWT-based authentication system
- **Performance**: Bundle analyzer, Lighthouse integration

## Getting Started

### Prerequisites

- Node.js 18.0 or higher
- npm or yarn package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neurashield-frontend.git
   cd neurashield-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```
   # Create a .env.local file with the following variables
   NEXT_PUBLIC_BLOCKCHAIN_API_URL=http://localhost:3001
   NEXT_PUBLIC_IPFS_GATEWAY=https://ipfs.io/ipfs/
   NEXT_PUBLIC_USE_MOCK_API=true  # Set to false for production
   ```

### Development

Start the development server:

```bash
npm run dev
```

The application will be available at http://localhost:3000.

### Testing

Run the test suite:

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage
```

### Performance Testing

The application includes several tools for performance testing and optimization:

```bash
# Analyze bundle size with @next/bundle-analyzer
npm run perf:analyze

# Run Google Lighthouse performance audit
npm run perf:lighthouse

# Analyze JavaScript source maps
npm run perf:bundle
```

Additionally, the application includes a built-in performance monitoring page at `/performance` that provides real-time metrics on Core Web Vitals and other performance indicators.

### Building for Production

Build the application for production:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

## Project Structure

```
neurashield-frontend/
├── app/              # Next.js App Router pages and layouts
├── components/       # Reusable UI components
├── lib/              # Utilities, hooks, and services
│   ├── api/          # API service clients
│   ├── auth/         # Authentication logic
│   ├── hooks/        # Custom React hooks
│   ├── services/     # Service implementations
│   └── types/        # TypeScript type definitions
├── public/           # Static assets
├── styles/           # Global styles
├── __tests__/        # Test files
└── types/            # Global type definitions
```

## Key Components

- **Dashboard**: Main interface for security monitoring
- **Threats**: Display and analysis of detected threats
- **Evidence**: Management of evidence stored on IPFS
- **Blockchain**: Viewing blockchain-verified security events
- **AI Models**: Configuration and performance of AI detection models
- **IPFS Storage**: Interface for managing distributed storage
- **Settings**: Application and user settings
- **Users**: User management interface
- **Performance**: Performance monitoring and optimization

## Mock Services

The application includes mock implementations of all services for development and testing purposes. To use mock services:

1. Set `NEXT_PUBLIC_USE_MOCK_API=true` in your environment variables
2. The mock services simulate network latency and provide realistic test data

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 