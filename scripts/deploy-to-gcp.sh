#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID=""
REGION="us-central1"
SERVICE_NAME="neurashield-threat-daemon"
DATA_API_URL=""
BLOCKCHAIN_API_URL=""
BLOCKCHAIN_ENABLED="true"
MONITORING_INTERVAL="5.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --data-api-url)
      DATA_API_URL="$2"
      shift 2
      ;;
    --blockchain-api-url)
      BLOCKCHAIN_API_URL="$2"
      shift 2
      ;;
    --blockchain-enabled)
      BLOCKCHAIN_ENABLED="$2"
      shift 2
      ;;
    --monitoring-interval)
      MONITORING_INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$PROJECT_ID" ]; then
  echo -e "${RED}Error: --project-id is required${NC}"
  exit 1
fi

if [ -z "$DATA_API_URL" ]; then
  echo -e "${RED}Error: --data-api-url is required${NC}"
  exit 1
fi

if [ -z "$BLOCKCHAIN_API_URL" ]; then
  echo -e "${RED}Error: --blockchain-api-url is required${NC}"
  exit 1
fi

# Ensure we're in the root directory of the project
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
  echo -e "${RED}Error: Google Cloud SDK (gcloud) is not installed${NC}"
  echo "Please install it from: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

# Check if user is authenticated with gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
  echo -e "${RED}Error: Not authenticated with gcloud${NC}"
  echo "Please run: gcloud auth login"
  exit 1
fi

echo -e "${GREEN}Deploying NeuraShield Threat Detection Daemon to GCP...${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service name: $SERVICE_NAME"

# Set the current project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo -e "${GREEN}Enabling required GCP APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com \
  containerregistry.googleapis.com monitoring.googleapis.com logging.googleapis.com

# Create a service account if it doesn't exist
SERVICE_ACCOUNT="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" &> /dev/null; then
  echo -e "${GREEN}Creating service account: $SERVICE_ACCOUNT${NC}"
  gcloud iam service-accounts create "$SERVICE_NAME" \
    --display-name="NeuraShield Threat Detection Daemon Service Account"
fi

# Grant necessary permissions to the service account
echo -e "${GREEN}Granting necessary IAM permissions...${NC}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/logging.logWriter"
  
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/monitoring.metricWriter"

# Trigger Cloud Build
echo -e "${GREEN}Triggering Cloud Build...${NC}"
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_DATA_API_URL="$DATA_API_URL",_BLOCKCHAIN_API_URL="$BLOCKCHAIN_API_URL",_BLOCKCHAIN_ENABLED="$BLOCKCHAIN_ENABLED",_SERVICE_ACCOUNT="$SERVICE_ACCOUNT" \
  .

echo -e "${GREEN}Deployment initiated successfully!${NC}"
echo "You can check the status of your deployment with:"
echo "gcloud builds list"
echo ""
echo "Once deployed, you can view your service at:"
echo "https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID" 