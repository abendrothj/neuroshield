#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p /logs /app/ai_models/models

# Start the Python application
exec python -m ai_models.main 