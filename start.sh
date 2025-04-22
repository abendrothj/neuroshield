#!/bin/bash

# NeuraShield Startup Script
# This script helps users start the NeuraShield system

# Function to display help message
display_help() {
    echo "NeuraShield - Advanced Network Threat Detection System"
    echo ""
    echo "Usage: ./start.sh [option]"
    echo ""
    echo "Options:"
    echo "  -a, --api        Start the API service"
    echo "  -d, --dashboard  Start the dashboard"
    echo "  -t, --traffic    Start the traffic processor (requires interface name)"
    echo "  -c, --check      Check environment and dependencies"
    echo "  -dc, --docker    Start all services using Docker Compose"
    echo "  -h, --help       Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./start.sh --api                 # Start the API service"
    echo "  ./start.sh --dashboard           # Start the dashboard"
    echo "  ./start.sh --traffic eth0        # Start traffic processor on eth0"
    echo "  ./start.sh --docker              # Start using Docker Compose"
    echo ""
}

# Function to check environment
check_environment() {
    echo "Checking NeuraShield environment..."
    
    # Check Python version
    if command -v python3 &>/dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2)
        echo "✓ Python version: $python_version"
    else
        echo "✗ Python 3 not found. Please install Python 3.8 or higher."
        return 1
    fi
    
    # Check TensorFlow
    if python3 -c "import tensorflow as tf; print(f'✓ TensorFlow version: {tf.__version__}')" 2>/dev/null; then
        :
    else
        echo "✗ TensorFlow not found. Please install requirements."
        return 1
    fi
    
    # Check if model exists
    model_path="models/multi_dataset/chained_transfer_improved/best_model.keras"
    if [ -f "$model_path" ]; then
        echo "✓ Model found: $model_path"
    else
        echo "✗ Model not found: $model_path"
        echo "  Please download or train the model first."
    fi
    
    # Check for required directories
    for dir in "logs" "features" "models"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "✓ Created directory: $dir"
        else
            echo "✓ Directory exists: $dir"
        fi
    done
    
    # Check Docker if docker option selected
    if [[ "$1" == "docker" ]]; then
        if command -v docker &>/dev/null; then
            docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
            echo "✓ Docker version: $docker_version"
        else
            echo "✗ Docker not found. Please install Docker."
            return 1
        fi
        
        if command -v docker-compose &>/dev/null; then
            dc_version=$(docker-compose --version | cut -d' ' -f3 | tr -d ',')
            echo "✓ Docker Compose version: $dc_version"
        else
            echo "✗ Docker Compose not found. Please install Docker Compose."
            return 1
        fi
    fi
    
    echo "Environment check complete."
    return 0
}

# Function to start the API service
start_api() {
    echo "Starting NeuraShield API service..."
    python3 api.py
}

# Function to start the dashboard
start_dashboard() {
    echo "Starting NeuraShield Dashboard..."
    streamlit run dashboard.py
}

# Function to start the traffic processor
start_traffic_processor() {
    if [ -z "$1" ]; then
        echo "Error: Interface name required for traffic processor."
        echo "Usage: ./start.sh --traffic <interface_name>"
        return 1
    fi
    
    echo "Starting NeuraShield Traffic Processor on interface $1..."
    python3 traffic_processor.py --interface "$1"
}

# Function to start with Docker Compose
start_docker() {
    echo "Starting NeuraShield with Docker Compose..."
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        echo "Error: docker-compose.yml not found."
        return 1
    fi
    
    # Build and start containers
    docker-compose up -d
    
    echo "NeuraShield services started in containers."
    echo "- API available at: http://localhost:8000"
    echo "- Dashboard available at: http://localhost:8501"
}

# Main script execution
if [ "$#" -eq 0 ]; then
    display_help
    exit 0
fi

case "$1" in
    -a|--api)
        check_environment
        start_api
        ;;
    -d|--dashboard)
        check_environment
        start_dashboard
        ;;
    -t|--traffic)
        check_environment
        start_traffic_processor "$2"
        ;;
    -c|--check)
        check_environment
        ;;
    -dc|--docker)
        check_environment "docker"
        start_docker
        ;;
    -h|--help)
        display_help
        ;;
    *)
        echo "Unknown option: $1"
        display_help
        exit 1
        ;;
esac

exit 0 