#!/bin/bash

# Script to configure the project for either development or production

MODE=${1:-dev}

# Create necessary directories
mkdir -p backend/logs
mkdir -p frontend/.next

echo "==== NeuraShield Environment Setup ===="

if [ "$MODE" == "dev" ] || [ "$MODE" == "development" ]; then
  echo "Setting up DEVELOPMENT environment..."
  
  # Create/update .env.local for frontend
  cat > frontend/.env.local << EOF
NEXT_PUBLIC_SOCKET_URL=http://localhost:3001
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:5000
EOF

  # Create/update .env for backend
  cat > backend/.env << EOF
# Server Configuration
PORT=3001
NODE_ENV=development

# API Endpoints
AI_SERVICE_URL=http://ai-service:5000

# Blockchain Configuration
CHANNEL_NAME=neurashield-channel
CONTRACT_NAME=neurashield
ORGANIZATION_ID=Org1
USER_ID=admin

# IPFS Configuration
IPFS_URL=http://ipfs:5001

# Logging
LOG_LEVEL=debug

# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=neurashield
DB_USER=neurauser
DB_PASSWORD=devpassword
EOF

  echo "Environment set to DEVELOPMENT"
  echo "To start your development environment, run: npm run start:dev"

elif [ "$MODE" == "prod" ] || [ "$MODE" == "production" ]; then
  echo "Setting up PRODUCTION environment..."
  
  # Generate secure passwords
  DB_PASSWORD=$(openssl rand -hex 12)
  GRAFANA_PASSWORD=$(openssl rand -hex 12)
  
  # Create/update .env.local for frontend
  cat > frontend/.env.local << EOF
NEXT_PUBLIC_SOCKET_URL=https://api.neurashield.com
NEXT_PUBLIC_API_URL=https://api.neurashield.com
NEXT_PUBLIC_AI_SERVICE_URL=https://api.neurashield.com/ai
EOF

  # Create/update .env for backend
  cat > backend/.env << EOF
# Server Configuration
PORT=3001
NODE_ENV=production

# API Endpoints
AI_SERVICE_URL=http://ai-service:5000

# Blockchain Configuration
CHANNEL_NAME=neurashield-channel
CONTRACT_NAME=neurashield
ORGANIZATION_ID=Org1
USER_ID=admin

# IPFS Configuration
IPFS_URL=http://ipfs:5001

# Logging
LOG_LEVEL=info

# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=neurashield
DB_USER=neurauser
DB_PASSWORD=${DB_PASSWORD}
EOF

  # Create .env in project root
  cat > .env << EOF
# Production environment variables
DB_USER=neurauser
DB_PASSWORD=${DB_PASSWORD}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
EOF

  echo "Environment set to PRODUCTION"
  echo "Generated secure passwords have been saved to your .env files"
  echo "IMPORTANT: Keep these passwords safe and back them up securely"
  echo "To start in production mode, run: npm run start:prod"
  
else
  echo "Invalid mode: $MODE"
  echo "Usage: $0 [dev|prod]"
  echo "  dev  - Configure for development (default)"
  echo "  prod - Configure for production"
  exit 1
fi

echo "Configuration completed successfully!" 