#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== NeuraShield Shutdown Script =====${NC}"

# Stop the server if running
if [ -f "/home/jub/Cursor/neurashield/backend/server.pid" ]; then
  echo -e "${YELLOW}Stopping NeuraShield server...${NC}"
  kill $(cat /home/jub/Cursor/neurashield/backend/server.pid) 2>/dev/null || true
  rm -f /home/jub/Cursor/neurashield/backend/server.pid
  echo -e "${GREEN}✓ Server stopped${NC}"
else
  echo -e "${YELLOW}Server is not running (no PID file found)${NC}"
fi

# Ask user if they want to stop the Fabric network
read -p "Do you want to stop the Fabric network as well? (y/n): " -n 1 -r STOP_FABRIC
echo ""

if [[ $STOP_FABRIC =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Stopping Fabric network...${NC}"
  cd /home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network
  ./network.sh down
  echo -e "${GREEN}✓ Fabric network stopped${NC}"
else
  echo -e "${YELLOW}Fabric network will continue running${NC}"
fi

echo -e "${GREEN}✓ NeuraShield has been shut down${NC}" 