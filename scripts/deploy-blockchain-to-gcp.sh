#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID="supple-defender-458307-i7"
REGION="us-west1"
SERVICE_NAME="neurashield-blockchain"
CPU="1"
MEMORY="2Gi"
VPC_CONNECTOR="neurashield-vpc-connector"

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
    --cpu)
      CPU="$2"
      shift 2
      ;;
    --memory)
      MEMORY="$2"
      shift 2
      ;;
    --vpc-connector)
      VPC_CONNECTOR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate project ID format
if ! [[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
  echo -e "${RED}Error: Invalid project ID format${NC}"
  echo "Project ID must be 6-30 characters long, start with a letter, and contain only lowercase letters, numbers, and hyphens"
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

# Check if project exists and user has access
if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
  echo -e "${RED}Error: Project $PROJECT_ID does not exist or you don't have access to it${NC}"
  exit 1
fi

echo -e "${GREEN}Deploying NeuraShield Blockchain Service to GCP...${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service name: $SERVICE_NAME"
echo "CPU: $CPU"
echo "Memory: $MEMORY"

# Set the current project
echo -e "${GREEN}Setting project to $PROJECT_ID...${NC}"
gcloud config set project "$PROJECT_ID" --quiet

# Enable required APIs
echo -e "${GREEN}Enabling required GCP APIs...${NC}"
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  containerregistry.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  vpcaccess.googleapis.com \
  --project="$PROJECT_ID" \
  --quiet

# Create a service account if it doesn't exist
SERVICE_ACCOUNT="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project="$PROJECT_ID" &> /dev/null; then
  echo -e "${GREEN}Creating service account: $SERVICE_ACCOUNT${NC}"
  gcloud iam service-accounts create "$SERVICE_NAME" \
    --display-name="NeuraShield Blockchain Service Account" \
    --project="$PROJECT_ID" \
    --quiet
fi

# Grant necessary permissions to the service account
echo -e "${GREEN}Granting necessary IAM permissions...${NC}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/logging.logWriter" \
  --quiet

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/monitoring.metricWriter" \
  --quiet

# Create VPC connector if it doesn't exist
if ! gcloud compute networks vpc-access connectors describe "$VPC_CONNECTOR" --region="$REGION" --project="$PROJECT_ID" &> /dev/null; then
  echo -e "${GREEN}Creating VPC connector: $VPC_CONNECTOR${NC}"
  gcloud compute networks vpc-access connectors create "$VPC_CONNECTOR" \
    --region="$REGION" \
    --network=default \
    --range=10.8.0.0/28 \
    --min-instances=2 \
    --max-instances=3 \
    --machine-type=e2-micro \
    --project="$PROJECT_ID" \
    --quiet
fi

# Trigger Cloud Build
echo -e "${GREEN}Triggering Cloud Build...${NC}"
gcloud builds submit --config cloudbuild-blockchain.yaml \
  --project="$PROJECT_ID" \
  --substitutions=_REGION="$REGION",_CPU="$CPU",_MEMORY="$MEMORY",_SERVICE_ACCOUNT="$SERVICE_ACCOUNT",_VPC_CONNECTOR="$VPC_CONNECTOR" \
  .

echo -e "${GREEN}Deployment initiated successfully!${NC}"
echo "You can check the status of your deployment with:"
echo "gcloud builds list --project=$PROJECT_ID"
echo ""
echo "Once deployed, you can view your service at:"
echo "https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID"
echo ""
echo "To get the service URL:"
echo "gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)'" 