#!/bin/bash

# Create necessary directories in the local environment 
mkdir -p ./logs ./models/models

# Start the Python application with the virtual environment Python
./models/venv/bin/python -m models.main 