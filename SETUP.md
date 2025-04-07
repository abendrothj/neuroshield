# NeuraShield Setup Guide

This guide helps you set up and run NeuraShield in both development and production environments.

## Getting Started

### Prerequisites

- Docker (latest version with `docker compose` command)
- Node.js 18.x or higher
- npm 9.x or higher
- bash shell

### First-time Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neurashield.git
   cd neurashield
   ```

2. Make all scripts executable:
   ```bash
   bash scripts/make-scripts-executable.sh
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

## Development Mode

In development mode, the system uses local development settings, enables debug logging, and mounts source code directories for hot-reloading.

1. Configure for development:
   ```bash
   npm run setup:dev
   ```

2. Start the development environment:
   ```bash
   npm run start:dev
   ```

3. Access services:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:3001
   - AI Service: http://localhost:5000
   - Grafana: http://localhost:3003 (admin/admin)

## Production Mode

When ready for production, use the production setup which includes secure passwords, optimized settings, and security hardening.

1. Configure for production:
   ```bash
   npm run setup:prod
   ```
   This will generate secure passwords and store them in `.env` files.

2. Start the production environment:
   ```bash
   npm run start:prod
   ```

3. Access services (if running locally):
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:3001
   - AI Service: http://localhost:5000
   - Grafana: http://localhost:3003 (credentials in .env)

## Kubernetes Deployment

To deploy to Kubernetes (for production):

1. Configure for production:
   ```bash
   npm run setup:prod
   ```

2. Deploy to Kubernetes:
   ```bash
   npm run deploy:k8s
   ```

3. Access via configured ingress URLs:
   - Frontend: https://app.neurashield.com
   - Backend API: https://api.neurashield.com
   - Monitoring: https://monitoring.neurashield.com

## Router Migration

NeuraShield is transitioning from Next.js Pages Router to App Router. During this transition, both routers will work.

Run the migration helper script:
```bash
npm run migrate:pages-to-app
```

This creates App Router equivalents for all Pages Router pages, allowing dual operation during migration.

## Testing

Run tests:
```bash
npm test                        # Run all tests
npm run test:websocket          # Test websocket functionality
npm run test:security:sql-injection  # Test SQL injection protection
npm run system-test             # Run comprehensive system tests
```

## Stopping Services

To stop all services:
```bash
npm run stop
```

## Troubleshooting

- **Database connection issues**: Check postgres container logs with `docker logs neurashield-postgres-1`
- **Backend startup failures**: Check logs with `docker logs neurashield-backend-1`
- **Frontend build issues**: Check frontend logs with `docker logs neurashield-frontend-1` 