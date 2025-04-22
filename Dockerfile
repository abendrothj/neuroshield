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
    # Create necessary directories with permissions
    mkdir -p /logs /app/ai_models/models && \
    # Fix Python module imports
    touch /app/__init__.py && \
    touch /app/ai_models/__init__.py && \
    cd /app/ai_models && \
    ln -sf threat_detection_model.py /app/ai_models/threat_detection_model.py && \
    # Set up the log file with proper permissions before switching user
    touch /logs/ai_metrics.log && \
    # Fix __init__.py and import issues
    sed -i 's/from ai_models.metrics import/try:\n    from ai_models.metrics import/g' /app/ai_models/threat_detection_model.py && \
    sed -i '/from ai_models.metrics import/a except ImportError:\n    print("Metrics module not available")\n    METRICS_AVAILABLE = False' /app/ai_models/threat_detection_model.py && \
    # Set all permissions as root
    chmod -R 777 /logs && \
    chmod 666 /logs/ai_metrics.log && \
    # Create non-root user
    groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appgroup:appuser /app && \
    chown -R appgroup:appuser /logs
# Copy model files separately to ensure they're included
COPY ai_models/models/threat_detection_20250403_212211 /app/ai_models/models/threat_detection_20250403_212211/
RUN chown -R appuser:appgroup /app/ai_models/models/threat_detection_20250403_212211 && \
    chmod -R 755 /app/ai_models/models/threat_detection_20250403_212211
# Copy and set up startup script
COPY ai_models/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh
WORKDIR /app/ai_models
ENV PYTHONPATH=/app
ENV USE_GPU=false
EXPOSE 5000
EXPOSE 8000
USER appuser
CMD ["/app/startup.sh"]

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    tcpdump \
    libpcap-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and scripts
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/features /app/models

# Expose ports for API and Dashboard
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    echo "Starting API service..."\n\
    cd /app && python api.py\n\
elif [ "$1" = "dashboard" ]; then\n\
    echo "Starting Dashboard..."\n\
    cd /app && streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0\n\
elif [ "$1" = "traffic" ]; then\n\
    echo "Starting Traffic Processor..."\n\
    cd /app && python traffic_processor.py --interface $2\n\
elif [ "$1" = "all" ]; then\n\
    echo "Starting all services..."\n\
    cd /app && python api.py & \n\
    cd /app && streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 & \n\
    wait\n\
else\n\
    echo "Available commands:"\n\
    echo "  api - Start the API service"\n\
    echo "  dashboard - Start the Dashboard"\n\
    echo "  traffic - Start the Traffic Processor (requires interface name)"\n\
    echo "  all - Start API and Dashboard"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["all"] 