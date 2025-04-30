FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/checkpoints /app/data

# Copy requirements file
COPY models/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY models/ /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_DIR=/app/logs
ENV METRICS_PORT=8000
ENV PORT=5000
ENV MODEL_PATH=/app/models/threat_detection_20250420_233903.keras
ENV BLOCKCHAIN_API_URL=http://blockchain-adapter:3000/api/v1/events
ENV BLOCKCHAIN_ENABLED=true
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Expose ports for API and Metrics
EXPOSE 5000
EXPOSE 8000

# Entry point script
RUN chmod +x /app/startup.sh

# Start the application
CMD ["/bin/bash", "/app/startup.sh"]

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1 