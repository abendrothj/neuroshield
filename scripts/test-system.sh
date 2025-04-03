#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Testing NeuraShield system components...${NC}"

# Get minikube IP
MINIKUBE_IP=$(minikube ip)
echo -e "${GREEN}Minikube IP: ${MINIKUBE_IP}${NC}"

# Test frontend health
echo -e "${YELLOW}Testing frontend health...${NC}"
curl -s http://neurashield.local/health | grep -q "healthy" && \
  echo -e "${GREEN}Frontend health check passed${NC}" || \
  (echo -e "${RED}Frontend health check failed${NC}" && exit 1)

# Test backend health
echo -e "${YELLOW}Testing backend health...${NC}"
curl -s http://neurashield.local/api/health | grep -q "healthy" && \
  echo -e "${GREEN}Backend health check passed${NC}" || \
  (echo -e "${RED}Backend health check failed${NC}" && exit 1)

# Test AI service health
echo -e "${YELLOW}Testing AI service health...${NC}"
curl -s http://neurashield.local/ai/health | grep -q "healthy" && \
  echo -e "${GREEN}AI service health check passed${NC}" || \
  (echo -e "${RED}AI service health check failed${NC}" && exit 1)

# Test AI analysis
echo -e "${YELLOW}Testing AI analysis...${NC}"
curl -s -X POST http://neurashield.local/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"data":{"feature1":0.1,"feature2":0.2,"feature3":0.3,"feature4":0.4,"feature5":0.5,"feature6":0.6,"feature7":0.7,"feature8":0.8,"feature9":0.9,"feature10":1.0}}' | \
  grep -q "threat_level" && \
  echo -e "${GREEN}AI analysis test passed${NC}" || \
  (echo -e "${RED}AI analysis test failed${NC}" && exit 1)

# Test blockchain logging
echo -e "${YELLOW}Testing blockchain logging...${NC}"
curl -s -X POST http://neurashield.local/api/events \
  -H "Content-Type: application/json" \
  -d '{"id":"test1","timestamp":"2024-04-03T12:00:00Z","type":"test","details":{"test":"data"}}' | \
  grep -q "success" && \
  echo -e "${GREEN}Blockchain logging test passed${NC}" || \
  (echo -e "${RED}Blockchain logging test failed${NC}" && exit 1)

# Test event retrieval
echo -e "${YELLOW}Testing event retrieval...${NC}"
curl -s http://neurashield.local/api/events/test1 | grep -q "test1" && \
  echo -e "${GREEN}Event retrieval test passed${NC}" || \
  (echo -e "${RED}Event retrieval test failed${NC}" && exit 1)

echo -e "${GREEN}All system tests passed successfully!${NC}"

# Show pod status
echo -e "${YELLOW}Pod Status:${NC}"
kubectl get pods -l app=neurashield -n neurashield 