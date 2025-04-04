#!/bin/bash

# NeuraShield Backup and Recovery Script
# This script handles automated backup and recovery procedures for the NeuraShield system

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default variables
BACKUP_DIR="/opt/neurashield/backups"
RETENTION_DAYS=30
BACKUP_COMPONENTS=("database" "models" "blockchain" "configs")
ACTION="backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESTORE_TIMESTAMP=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --backup)
      ACTION="backup"
      shift
      ;;
    --restore)
      ACTION="restore"
      shift
      ;;
    --timestamp)
      RESTORE_TIMESTAMP="$2"
      shift 2
      ;;
    --components)
      IFS=',' read -ra BACKUP_COMPONENTS <<< "$2"
      shift 2
      ;;
    --help)
      echo "Usage: backup-recovery.sh [options]"
      echo "Options:"
      echo "  --backup               Perform backup (default)"
      echo "  --restore              Perform restore"
      echo "  --timestamp TIMESTAMP  Specify timestamp for restore (required for restore)"
      echo "  --components COMP1,COMP2  Specify components to backup/restore (default: all)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Verify requirements for restore
if [[ "$ACTION" == "restore" && -z "$RESTORE_TIMESTAMP" ]]; then
  echo -e "${RED}Error: Restore requires --timestamp parameter${NC}"
  exit 1
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Backup function for database
backup_database() {
  echo -e "${YELLOW}Backing up database...${NC}"
  kubectl exec -it $(kubectl get pods -l app=mongodb -o jsonpath="{.items[0].metadata.name}") -- \
    mongodump --out=/tmp/mongodb_backup
  
  kubectl cp $(kubectl get pods -l app=mongodb -o jsonpath="{.items[0].metadata.name}"):/tmp/mongodb_backup \
    "${BACKUP_DIR}/${TIMESTAMP}/mongodb_backup"
  
  echo -e "${GREEN}Database backup completed!${NC}"
}

# Backup function for AI models
backup_models() {
  echo -e "${YELLOW}Backing up AI models...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/models"
  
  kubectl cp $(kubectl get pods -l app=ai-service -o jsonpath="{.items[0].metadata.name}"):/app/models \
    "${BACKUP_DIR}/${TIMESTAMP}/models"
  
  echo -e "${GREEN}AI models backup completed!${NC}"
}

# Backup function for blockchain data
backup_blockchain() {
  echo -e "${YELLOW}Backing up blockchain data...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/blockchain"
  
  kubectl cp $(kubectl get pods -l app=blockchain -o jsonpath="{.items[0].metadata.name}"):/app/blockchain \
    "${BACKUP_DIR}/${TIMESTAMP}/blockchain"
  
  echo -e "${GREEN}Blockchain data backup completed!${NC}"
}

# Backup function for configuration files
backup_configs() {
  echo -e "${YELLOW}Backing up configuration files...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/configs"
  
  # Export all configmaps and secrets
  kubectl get configmaps -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configs/configmaps.yaml"
  kubectl get secrets -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configs/secrets.yaml"
  
  echo -e "${GREEN}Configuration backup completed!${NC}"
}

# Restore function for database
restore_database() {
  echo -e "${YELLOW}Restoring database from backup...${NC}"
  
  # Copy backup files to MongoDB pod
  kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/mongodb_backup" \
    $(kubectl get pods -l app=mongodb -o jsonpath="{.items[0].metadata.name}"):/tmp/mongodb_backup
  
  # Restore from backup
  kubectl exec -it $(kubectl get pods -l app=mongodb -o jsonpath="{.items[0].metadata.name}") -- \
    mongorestore /tmp/mongodb_backup
  
  echo -e "${GREEN}Database restore completed!${NC}"
}

# Restore function for AI models
restore_models() {
  echo -e "${YELLOW}Restoring AI models from backup...${NC}"
  
  kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models" \
    $(kubectl get pods -l app=ai-service -o jsonpath="{.items[0].metadata.name}"):/app/models
  
  # Restart AI service to load new models
  kubectl rollout restart deployment ai-service
  
  echo -e "${GREEN}AI models restore completed!${NC}"
}

# Restore function for blockchain data
restore_blockchain() {
  echo -e "${YELLOW}Restoring blockchain data from backup...${NC}"
  
  kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/blockchain" \
    $(kubectl get pods -l app=blockchain -o jsonpath="{.items[0].metadata.name}"):/app/blockchain
  
  # Restart blockchain service
  kubectl rollout restart deployment blockchain
  
  echo -e "${GREEN}Blockchain data restore completed!${NC}"
}

# Restore function for configuration files
restore_configs() {
  echo -e "${YELLOW}Restoring configuration files from backup...${NC}"
  
  # Apply configmaps and secrets
  kubectl apply -f "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/configmaps.yaml"
  kubectl apply -f "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/secrets.yaml"
  
  echo -e "${GREEN}Configuration restore completed!${NC}"
}

# Clean up old backups
cleanup_old_backups() {
  echo -e "${YELLOW}Cleaning up backups older than ${RETENTION_DAYS} days...${NC}"
  find "$BACKUP_DIR" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null || true
  echo -e "${GREEN}Cleanup completed!${NC}"
}

# Perform selected action
if [[ "$ACTION" == "backup" ]]; then
  echo -e "${YELLOW}Starting backup process with timestamp: ${TIMESTAMP}${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"
  
  for component in "${BACKUP_COMPONENTS[@]}"; do
    case "$component" in
      "database")
        backup_database
        ;;
      "models")
        backup_models
        ;;
      "blockchain")
        backup_blockchain
        ;;
      "configs")
        backup_configs
        ;;
      *)
        echo -e "${RED}Unknown component: $component${NC}"
        ;;
    esac
  done
  
  cleanup_old_backups
  echo -e "${GREEN}Backup process completed successfully!${NC}"
  
elif [[ "$ACTION" == "restore" ]]; then
  echo -e "${YELLOW}Starting restore process from timestamp: ${RESTORE_TIMESTAMP}${NC}"
  
  # Check if backup exists
  if [[ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}" ]]; then
    echo -e "${RED}Error: Backup with timestamp ${RESTORE_TIMESTAMP} not found!${NC}"
    exit 1
  fi
  
  for component in "${BACKUP_COMPONENTS[@]}"; do
    case "$component" in
      "database")
        restore_database
        ;;
      "models")
        restore_models
        ;;
      "blockchain")
        restore_blockchain
        ;;
      "configs")
        restore_configs
        ;;
      *)
        echo -e "${RED}Unknown component: $component${NC}"
        ;;
    esac
  done
  
  echo -e "${GREEN}Restore process completed successfully!${NC}"
fi 