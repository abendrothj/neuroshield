#!/bin/bash

# NeuraShield Setup Script
# This script handles both local and AI environment setup

set -e

# Configuration
AI_MODELS_DIR="models"
BACKEND_DIR="backend"
BLOCKCHAIN_DIR="blockchain"
OUTPUT_DIR="output"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Print with color
print() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    print "All prerequisites met"
}

# Setup Python environment
setup_python_env() {
    print "Setting up Python environment..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    
    print "Python environment setup complete"
}

# Setup Node.js environment
setup_node_env() {
    print "Setting up Node.js environment..."
    
    cd $BACKEND_DIR
    npm install
    cd ..
    
    print "Node.js environment setup complete"
}

# Setup AI models
setup_ai_models() {
    print "Setting up AI models..."
    
    mkdir -p $AI_MODELS_DIR
    mkdir -p $OUTPUT_DIR
    
    # Download pre-trained models if needed
    if [ ! -d "$AI_MODELS_DIR/pretrained" ]; then
        print "Downloading pre-trained models..."
        # Add your model download commands here
    fi
    
    print "AI models setup complete"
}

# Setup blockchain environment
setup_blockchain() {
    print "Setting up blockchain environment..."
    
    cd $BLOCKCHAIN_DIR
    ./bootstrap.sh
    cd ..
    
    print "Blockchain environment setup complete"
}

# Main setup function
main() {
    print "Starting NeuraShield setup..."
    
    check_prerequisites
    setup_python_env
    setup_node_env
    setup_ai_models
    setup_blockchain
    
    print "NeuraShield setup complete!"
    print "To activate the Python environment, run: source venv/bin/activate"
}

# Run main function
main 