#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== NeuraShield Cleanup Script =====${NC}"

# Directory paths
PROJECT_ROOT="/home/jub/Cursor/neurashield"
LOGS_DIR="${PROJECT_ROOT}/logs"
TEMP_DIR="${PROJECT_ROOT}/temp"
CHAINCODE_TEMP="${PROJECT_ROOT}/backend/chaincode/simple-pkg"
BACKEND_LOGS="${PROJECT_ROOT}/backend/logs"

# Function to safely clean a directory without removing it
clean_dir() {
  local dir=$1
  if [ -d "$dir" ]; then
    echo -e "${YELLOW}Cleaning directory: ${dir}${NC}"
    find "$dir" -type f -not -name '.gitkeep' -delete
    echo -e "${GREEN}✓ Cleaned${NC}"
  else
    echo -e "${YELLOW}Directory does not exist: ${dir} (skipping)${NC}"
  fi
}

# Function to rotate log files
rotate_logs() {
  local dir=$1
  if [ -d "$dir" ]; then
    echo -e "${YELLOW}Rotating logs in: ${dir}${NC}"
    find "$dir" -type f -name "*.log" -size +10M | while read file; do
      if [ ! -f "${file}.1" ]; then
        mv "$file" "${file}.1"
        touch "$file"
        echo -e "${GREEN}✓ Rotated: ${file}${NC}"
      fi
    done
  fi
}

# Clean up logs
echo -e "${YELLOW}Cleaning up log files...${NC}"
clean_dir "$LOGS_DIR"
rotate_logs "$BACKEND_LOGS"

# Clean up temporary files
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
clean_dir "$TEMP_DIR"
[ -d "$CHAINCODE_TEMP" ] && rm -rf "$CHAINCODE_TEMP" && echo -e "${GREEN}✓ Removed temporary chaincode package${NC}"

# Clean up docker build cache (optional)
read -p "Do you want to clean Docker build cache as well? (y/n): " -n 1 -r CLEAN_DOCKER
echo ""

if [[ $CLEAN_DOCKER =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Cleaning Docker build cache...${NC}"
  docker system prune -f
  echo -e "${GREEN}✓ Docker cache cleaned${NC}"
else
  echo -e "${YELLOW}Skipping Docker cache cleanup${NC}"
fi

# Create needed directories that might have been deleted
mkdir -p "$LOGS_DIR" "$TEMP_DIR" "$BACKEND_LOGS"

echo -e "${GREEN}✓ Cleanup completed${NC}" 