# Multi-stage build for NeuraShield components

# Backend build
FROM node:18.19.1-alpine AS backend-build
WORKDIR /app/backend
COPY backend/package*.json ./
RUN npm ci
COPY backend/ .

# Frontend build
FROM node:18.19.1-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN npm install -g pnpm
RUN pnpm install
# Set up app directory properly
RUN mkdir -p /app/frontend/app
COPY frontend/ .
RUN pnpm install
RUN pnpm run build

# AI Service build
FROM python:3.9-slim AS ai-build
WORKDIR /app/ai_models
COPY ai_models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai_models/ .

# Backend runtime
FROM node:18.19.1-alpine AS backend
WORKDIR /app
COPY --from=backend-build /app/backend ./backend
WORKDIR /app/backend
ENV NODE_ENV=production
# Update the ipfs-http-client dependency
RUN npm uninstall ipfs-http-client && npm install ipfs-http-client@56.0.2
# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
RUN chown -R appuser:appgroup /app
USER appuser
EXPOSE 3001
CMD ["node", "src/server.js"]

# Frontend runtime
FROM node:18.19.1-alpine AS frontend
WORKDIR /app
COPY --from=frontend-build /app/frontend/package.json ./frontend/
COPY --from=frontend-build /app/frontend/next.config.mjs ./frontend/
COPY --from=frontend-build /app/frontend/public ./frontend/public
COPY --from=frontend-build /app/frontend/.next ./frontend/.next
COPY --from=frontend-build /app/frontend/node_modules ./frontend/node_modules
WORKDIR /app/frontend
ENV NODE_ENV=production
# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
RUN chown -R appuser:appgroup /app
USER appuser
EXPOSE 3000
CMD ["npx", "next", "start"]

# AI service runtime
FROM python:3.9-slim AS ai-service
WORKDIR /app
COPY --from=ai-build /app/ai_models ./ai_models
# Install dependencies and set up directories
COPY ai_models/requirements.txt ./ai_models/
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r ./ai_models/requirements.txt && \
    mkdir -p /logs /app/ai_models/models && \
    groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /logs /app/ai_models/models && \
    chmod -R 755 /logs /app/ai_models/models
# Copy model files separately to ensure they're included
COPY ai_models/models/threat_detection_20250403_212211 /app/ai_models/models/threat_detection_20250403_212211/
RUN chown -R appuser:appgroup /app/ai_models/models/threat_detection_20250403_212211 && \
    chmod -R 755 /app/ai_models/models/threat_detection_20250403_212211
# Copy and set up startup script
COPY ai_models/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh
WORKDIR /app/ai_models
ENV PYTHONPATH=/app
EXPOSE 5000
EXPOSE 8000
USER appuser
CMD ["/app/startup.sh"] 