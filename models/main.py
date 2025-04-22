"""
Main entry point for the NeuraShield AI service.
"""

import os
import uvicorn
import logging

# Get log directory from environment variable or use current directory
LOG_DIR = os.environ.get('LOG_DIR', '.')
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, 'ai_service.log')

from ai_models.metrics import start_metrics_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Start the AI service and metrics server"""
    try:
        # Start metrics server on port 8000
        metrics_port = int(os.environ.get("METRICS_PORT", 8000))
        start_metrics_server(port=metrics_port)
        logger.info(f"Metrics server started on port {metrics_port}")
        
        # Start the API server
        api_port = int(os.environ.get("PORT", 5000))
        logger.info(f"Starting API server on port {api_port}")
        
        # Use uvicorn to run the FastAPI application
        uvicorn.run(
            "ai_models.api:app",
            host="0.0.0.0",
            port=api_port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting AI service: {str(e)}")
        raise

if __name__ == "__main__":
    main() 