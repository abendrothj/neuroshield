#!/bin/bash
# Script to apply model updates to a deployment

# Exit immediately if a command exits with a non-zero status
set -e

# Display commands being executed
set -x

# Default values
MODEL_PATH=""
ENVIRONMENT="production"
RESTART_SERVICES=true
BACKUP=true
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model-path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --environment)
      ENVIRONMENT="$2"
      shift
      shift
      ;;
    --no-restart)
      RESTART_SERVICES=false
      shift
      ;;
    --no-backup)
      BACKUP=false
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model-path PATH       Path to the new model (required)"
      echo "  --environment ENV       Target environment [production|staging|development] (default: production)"
      echo "  --no-restart            Don't restart services after update"
      echo "  --no-backup             Don't backup current model before update"
      echo "  --force                 Force update even if validation fails"
      echo "  -h, --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model-path is required"
  exit 1
fi

# Setup paths based on environment
case "$ENVIRONMENT" in
  production)
    TARGET_DIR="/app/ai_models/models"
    CONFIG_FILE="/app/ai_models/.env"
    ;;
  staging)
    TARGET_DIR="/app/staging/ai_models/models"
    CONFIG_FILE="/app/staging/ai_models/.env"
    ;;
  development)
    TARGET_DIR="./ai_models/models"
    CONFIG_FILE="./ai_models/.env"
    ;;
  *)
    echo "Error: Invalid environment '$ENVIRONMENT'"
    exit 1
    ;;
esac

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model directory not found at $MODEL_PATH"
  exit 1
fi

# Get current date/time for versioning
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOYMENT_NAME="threat_detection_${TIMESTAMP}"
DEPLOYMENT_PATH="${TARGET_DIR}/${DEPLOYMENT_NAME}"

# Create backup if enabled
if [ "$BACKUP" = true ]; then
  CURRENT_MODEL="${TARGET_DIR}/threat_detection_latest"
  if [ -L "$CURRENT_MODEL" ]; then
    CURRENT_TARGET=$(readlink -f "$CURRENT_MODEL")
    BACKUP_DIR="${TARGET_DIR}/backup_${TIMESTAMP}"
    echo "Backing up current model to $BACKUP_DIR"
    cp -r "$CURRENT_TARGET" "$BACKUP_DIR"
  fi
fi

# Create deployment directory
echo "Creating deployment directory at $DEPLOYMENT_PATH"
mkdir -p "$DEPLOYMENT_PATH"

# Copy model files
echo "Copying model files from $MODEL_PATH to $DEPLOYMENT_PATH"
cp -r "${MODEL_PATH}/." "$DEPLOYMENT_PATH/"

# Validate model
echo "Validating model..."
if [ -f "${DEPLOYMENT_PATH}/model.keras" ] && [ -f "${DEPLOYMENT_PATH}/metadata.json" ]; then
  echo "Model files validated successfully"
else
  echo "Error: Model validation failed, missing required files"
  if [ "$FORCE" = false ]; then
    echo "Update aborted. Use --force to override validation"
    # Cleanup
    rm -rf "$DEPLOYMENT_PATH"
    exit 1
  else
    echo "Proceeding with update (--force enabled)"
  fi
fi

# Update symlink to point to the new model
echo "Updating symlink to new model"
ln -sfn "$DEPLOYMENT_PATH" "${TARGET_DIR}/threat_detection_latest"

# Update environment file
if [ -f "$CONFIG_FILE" ]; then
  echo "Updating environment file at $CONFIG_FILE"
  # If MODEL_PATH line exists, replace it, otherwise add it
  if grep -q "^MODEL_PATH=" "$CONFIG_FILE"; then
    sed -i "s|^MODEL_PATH=.*|MODEL_PATH=${TARGET_DIR}/threat_detection_latest|" "$CONFIG_FILE"
  else
    echo "MODEL_PATH=${TARGET_DIR}/threat_detection_latest" >> "$CONFIG_FILE"
  fi
fi

# Restart services if enabled
if [ "$RESTART_SERVICES" = true ]; then
  echo "Restarting services..."
  if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "staging" ]; then
    # For production/staging, use Docker commands
    docker-compose restart ai-service
  else
    # For development, use simpler approach
    echo "Development environment: Please restart the services manually"
  fi
fi

echo "Model update completed successfully!"
echo "New model deployed at: ${DEPLOYMENT_PATH}"
echo "Symlink updated at: ${TARGET_DIR}/threat_detection_latest" 