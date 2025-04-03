#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if required Docker images exist
echo "Checking Docker images..."
required_images=("neurashield-frontend:latest" "neurashield-backend:latest" "neurashield-ai:latest")
missing_images=()

for image in "${required_images[@]}"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        missing_images+=("$image")
    fi
done

if [ ${#missing_images[@]} -ne 0 ]; then
    echo -e "${RED}Missing required Docker images:${NC}"
    for image in "${missing_images[@]}"; do
        echo "  - $image"
    done
    echo -e "${YELLOW}Please build the required images before deployment.${NC}"
    exit 1
fi

# Check if ConfigMap exists
echo "Checking ConfigMap..."
if ! kubectl get configmap neurashield-config >/dev/null 2>&1; then
    echo -e "${RED}ConfigMap 'neurashield-config' not found.${NC}"
    echo -e "${YELLOW}Please create the ConfigMap before deployment.${NC}"
    exit 1
fi

# Check if secrets exist
echo "Checking secrets..."
if ! kubectl get secret neurashield-secrets >/dev/null 2>&1; then
    echo -e "${RED}Secret 'neurashield-secrets' not found.${NC}"
    echo -e "${YELLOW}Please create the secrets before deployment.${NC}"
    exit 1
fi

echo -e "${GREEN}All prerequisites satisfied!${NC}" 