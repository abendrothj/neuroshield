#!/bin/bash

# NeuraShield Backup and Recovery Script
# This script handles automated backup and recovery procedures for the NeuraShield system
# Compatible with both Kubernetes and Docker Compose environments

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default variables
BACKUP_DIR=${BACKUP_DIR:-"/opt/neurashield/backups"}
RETENTION_DAYS=${RETENTION_DAYS:-30}
BACKUP_COMPONENTS=("database" "models" "blockchain" "configs" "ipfs")
ACTION="backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESTORE_TIMESTAMP=""
ENVIRONMENT="docker" # Default to docker, will detect k8s if running there

# Detect environment (Kubernetes or Docker Compose)
if [ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]; then
    ENVIRONMENT="kubernetes"
    echo "Detected Kubernetes environment"
else
    echo "Detected Docker Compose environment"
fi

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
    --verify)
      ACTION="verify"
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
    --environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: backup-recovery.sh [options]"
      echo "Options:"
      echo "  --backup               Perform backup (default)"
      echo "  --restore              Perform restore"
      echo "  --verify               Verify backup integrity"
      echo "  --timestamp TIMESTAMP  Specify timestamp for restore (required for restore)"
      echo "  --components COMP1,COMP2  Specify components to backup/restore (default: all)"
      echo "  --environment docker|kubernetes  Force environment (auto-detected by default)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Verify requirements for restore or verify
if [[ "$ACTION" == "restore" && -z "$RESTORE_TIMESTAMP" ]]; then
  echo -e "${RED}Error: Restore requires --timestamp parameter${NC}"
  exit 1
fi

if [[ "$ACTION" == "verify" && -z "$RESTORE_TIMESTAMP" ]]; then
  # For verification, use the latest backup if not specified
  LATEST_BACKUP=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -n 1)
  if [[ -n "$LATEST_BACKUP" ]]; then
    RESTORE_TIMESTAMP=$(basename "$LATEST_BACKUP")
    echo -e "${YELLOW}No timestamp specified, using latest backup: ${RESTORE_TIMESTAMP}${NC}"
  else
    echo -e "${RED}Error: No backups found to verify${NC}"
    exit 1
  fi
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

#
# Docker Compose specific functions
#

# Backup function for Docker database
docker_backup_database() {
  echo -e "${YELLOW}Backing up database...${NC}"
  # Assuming MongoDB is used and data is in the volume
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/mongodb_backup"
  
  if [ -d "/app/backend/data/db" ]; then
    cp -r /app/backend/data/db/* "${BACKUP_DIR}/${TIMESTAMP}/mongodb_backup/"
  fi
  
  echo -e "${GREEN}Database backup completed!${NC}"
}

# Backup function for Docker AI models
docker_backup_models() {
  echo -e "${YELLOW}Backing up AI models...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/models"
  
  if [ -d "/app/ai_models/models" ]; then
    cp -r /app/ai_models/models/* "${BACKUP_DIR}/${TIMESTAMP}/models/"
  fi
  
  echo -e "${GREEN}AI models backup completed!${NC}"
}

# Backup function for Docker blockchain data
docker_backup_blockchain() {
  echo -e "${YELLOW}Backing up blockchain data...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/blockchain"
  
  if [ -d "/app/backend/data/blockchain" ]; then
    cp -r /app/backend/data/blockchain/* "${BACKUP_DIR}/${TIMESTAMP}/blockchain/"
  fi
  
  echo -e "${GREEN}Blockchain data backup completed!${NC}"
}

# Backup function for Docker IPFS data
docker_backup_ipfs() {
  echo -e "${YELLOW}Backing up IPFS data...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/ipfs"
  
  if [ -d "/data/ipfs" ]; then
    cp -r /data/ipfs/* "${BACKUP_DIR}/${TIMESTAMP}/ipfs/"
  fi
  
  echo -e "${GREEN}IPFS data backup completed!${NC}"
}

# Backup function for Docker configuration files
docker_backup_configs() {
  echo -e "${YELLOW}Backing up configuration files...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/configs"
  
  # Environment variables
  env > "${BACKUP_DIR}/${TIMESTAMP}/configs/environment.env"
  
  # Copy any config files from volumes
  if [ -d "/app/backend/config" ]; then
    mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/configs/backend"
    cp -r /app/backend/config/* "${BACKUP_DIR}/${TIMESTAMP}/configs/backend/"
  fi
  
  if [ -d "/app/ai_models/config" ]; then
    mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/configs/ai_models"
    cp -r /app/ai_models/config/* "${BACKUP_DIR}/${TIMESTAMP}/configs/ai_models/"
  fi
  
  echo -e "${GREEN}Configuration backup completed!${NC}"
}

# Restore function for Docker database
docker_restore_database() {
  echo -e "${YELLOW}Restoring database from backup...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/mongodb_backup" ]; then
    # Clear existing data first (optional)
    # rm -rf /app/backend/data/db/*
    
    # Restore the data
    cp -r "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/mongodb_backup/"* /app/backend/data/db/
  else
    echo -e "${RED}Database backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Database restore completed!${NC}"
}

# Restore function for Docker AI models
docker_restore_models() {
  echo -e "${YELLOW}Restoring AI models from backup...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models" ]; then
    # Clear existing models first (optional)
    # rm -rf /app/ai_models/models/*
    
    # Restore the models
    cp -r "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models/"* /app/ai_models/models/
  else
    echo -e "${RED}AI models backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}AI models restore completed!${NC}"
}

# Restore function for Docker blockchain data
docker_restore_blockchain() {
  echo -e "${YELLOW}Restoring blockchain data from backup...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/blockchain" ]; then
    # Clear existing data first (optional)
    # rm -rf /app/backend/data/blockchain/*
    
    # Restore the data
    cp -r "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/blockchain/"* /app/backend/data/blockchain/
  else
    echo -e "${RED}Blockchain backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Blockchain data restore completed!${NC}"
}

# Restore function for Docker IPFS data
docker_restore_ipfs() {
  echo -e "${YELLOW}Restoring IPFS data from backup...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/ipfs" ]; then
    # Clear existing data first (optional)
    # rm -rf /data/ipfs/*
    
    # Restore the data
    cp -r "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/ipfs/"* /data/ipfs/
  else
    echo -e "${RED}IPFS backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}IPFS data restore completed!${NC}"
}

# Restore function for Docker configuration files
docker_restore_configs() {
  echo -e "${YELLOW}Restoring configuration files from backup...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs" ]; then
    # Restore backend configs
    if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/backend" ]; then
      cp -r "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/backend/"* /app/backend/config/
    fi
    
    # Restore AI model configs
    if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/ai_models" ]; then
      cp -r "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/ai_models/"* /app/ai_models/config/
    fi
  else
    echo -e "${RED}Configuration backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Configuration restore completed!${NC}"
}

# Verify functions

# Verify backup integrity
verify_backup() {
  local component=$1
  local status=0
  
  echo -e "${YELLOW}Verifying ${component} backup integrity...${NC}"
  
  case "$component" in
    "database")
      if [ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/mongodb_backup" ]; then
        echo -e "${RED}Database backup is missing or incomplete${NC}"
        status=1
      else
        echo -e "${GREEN}Database backup verification passed${NC}"
      fi
      ;;
    "models")
      if [ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models" ]; then
        echo -e "${RED}AI models backup is missing or incomplete${NC}"
        status=1
      else
        # Check for at least one model file
        if [ -z "$(ls -A "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models" 2>/dev/null)" ]; then
          echo -e "${RED}AI models backup directory is empty${NC}"
          status=1
        else
          echo -e "${GREEN}AI models backup verification passed${NC}"
        fi
      fi
      ;;
    "blockchain")
      if [ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/blockchain" ]; then
        echo -e "${RED}Blockchain backup is missing or incomplete${NC}"
        status=1
      else
        echo -e "${GREEN}Blockchain backup verification passed${NC}"
      fi
      ;;
    "ipfs")
      if [ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/ipfs" ]; then
        echo -e "${RED}IPFS backup is missing or incomplete${NC}"
        status=1
      else
        echo -e "${GREEN}IPFS backup verification passed${NC}"
      fi
      ;;
    "configs")
      if [ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs" ]; then
        echo -e "${RED}Configuration backup is missing or incomplete${NC}"
        status=1
      else
        echo -e "${GREEN}Configuration backup verification passed${NC}"
      fi
      ;;
    *)
      echo -e "${RED}Unknown component: $component${NC}"
      status=1
      ;;
  esac
  
  return $status
}

# Clean up old backups
cleanup_old_backups() {
  echo -e "${YELLOW}Cleaning up backups older than ${RETENTION_DAYS} days...${NC}"
  find "$BACKUP_DIR" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null || true
  echo -e "${GREEN}Cleanup completed!${NC}"
}

# Kubernetes specific functions
#

# Backup function for Kubernetes database
k8s_backup_database() {
  echo -e "${YELLOW}Backing up database in Kubernetes environment...${NC}"
  
  if kubectl get pods -l app=mongodb -n neurashield &>/dev/null; then
    POD=$(kubectl get pods -l app=mongodb -n neurashield -o jsonpath="{.items[0].metadata.name}")
    kubectl exec -n neurashield $POD -- mongodump --out=/tmp/mongodb_backup
    kubectl cp -n neurashield $POD:/tmp/mongodb_backup "${BACKUP_DIR}/${TIMESTAMP}/mongodb_backup"
    kubectl exec -n neurashield $POD -- rm -rf /tmp/mongodb_backup
  else
    echo -e "${RED}MongoDB pod not found in Kubernetes, skipping database backup${NC}"
  fi
  
  echo -e "${GREEN}Database backup completed!${NC}"
}

# Backup function for Kubernetes AI models
k8s_backup_models() {
  echo -e "${YELLOW}Backing up AI models in Kubernetes environment...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/models"
  
  if kubectl get pvc ai-models-pvc -n neurashield &>/dev/null; then
    # We need to access the PVC through a pod to copy data
    echo "Creating temporary pod to access AI models PVC..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: backup-models-pod
  namespace: neurashield
spec:
  containers:
  - name: backup-container
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: models-volume
      mountPath: /models
  volumes:
  - name: models-volume
    persistentVolumeClaim:
      claimName: ai-models-pvc
EOF
    
    # Wait for pod to be ready
    kubectl wait --for=condition=Ready pod/backup-models-pod -n neurashield --timeout=60s
    
    # Copy the data
    kubectl cp -n neurashield backup-models-pod:/models "${BACKUP_DIR}/${TIMESTAMP}/models"
    
    # Clean up
    kubectl delete pod backup-models-pod -n neurashield
  else
    echo -e "${RED}AI models PVC not found in Kubernetes, skipping models backup${NC}"
  fi
  
  echo -e "${GREEN}AI models backup completed!${NC}"
}

# Backup function for Kubernetes blockchain data
k8s_backup_blockchain() {
  echo -e "${YELLOW}Backing up blockchain data in Kubernetes environment...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/blockchain"
  
  if kubectl get pvc blockchain-pvc -n neurashield &>/dev/null; then
    # We need to access the PVC through a pod to copy data
    echo "Creating temporary pod to access blockchain PVC..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: backup-blockchain-pod
  namespace: neurashield
spec:
  containers:
  - name: backup-container
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: blockchain-volume
      mountPath: /blockchain
  volumes:
  - name: blockchain-volume
    persistentVolumeClaim:
      claimName: blockchain-pvc
EOF
    
    # Wait for pod to be ready
    kubectl wait --for=condition=Ready pod/backup-blockchain-pod -n neurashield --timeout=60s
    
    # Copy the data
    kubectl cp -n neurashield backup-blockchain-pod:/blockchain "${BACKUP_DIR}/${TIMESTAMP}/blockchain"
    
    # Clean up
    kubectl delete pod backup-blockchain-pod -n neurashield
  else
    echo -e "${RED}Blockchain PVC not found in Kubernetes, skipping blockchain backup${NC}"
  fi
  
  echo -e "${GREEN}Blockchain data backup completed!${NC}"
}

# Backup function for Kubernetes IPFS data
k8s_backup_ipfs() {
  echo -e "${YELLOW}Backing up IPFS data in Kubernetes environment...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/ipfs"
  
  if kubectl get pvc ipfs-pvc -n neurashield &>/dev/null; then
    # We need to access the PVC through a pod to copy data
    echo "Creating temporary pod to access IPFS PVC..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: backup-ipfs-pod
  namespace: neurashield
spec:
  containers:
  - name: backup-container
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: ipfs-volume
      mountPath: /ipfs
  volumes:
  - name: ipfs-volume
    persistentVolumeClaim:
      claimName: ipfs-pvc
EOF
    
    # Wait for pod to be ready
    kubectl wait --for=condition=Ready pod/backup-ipfs-pod -n neurashield --timeout=60s
    
    # Copy the data
    kubectl cp -n neurashield backup-ipfs-pod:/ipfs "${BACKUP_DIR}/${TIMESTAMP}/ipfs"
    
    # Clean up
    kubectl delete pod backup-ipfs-pod -n neurashield
  else
    echo -e "${RED}IPFS PVC not found in Kubernetes, skipping IPFS backup${NC}"
  fi
  
  echo -e "${GREEN}IPFS data backup completed!${NC}"
}

# Backup function for Kubernetes configuration files
k8s_backup_configs() {
  echo -e "${YELLOW}Backing up configuration files in Kubernetes environment...${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}/configs"
  
  # Backup configmaps
  echo "Backing up ConfigMaps..."
  kubectl get configmaps -n neurashield -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configs/configmaps.yaml"
  
  # Backup secrets (note: this will include the Secret's data in base64 format)
  echo "Backing up Secrets..."
  kubectl get secrets -n neurashield -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configs/secrets.yaml"
  
  # Backup deployments
  echo "Backing up Deployments..."
  kubectl get deployments -n neurashield -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configs/deployments.yaml"
  
  # Backup services
  echo "Backing up Services..."
  kubectl get services -n neurashield -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configs/services.yaml"
  
  echo -e "${GREEN}Configuration backup completed!${NC}"
}

# Restore function for Kubernetes database
k8s_restore_database() {
  echo -e "${YELLOW}Restoring database in Kubernetes environment...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/mongodb_backup" ]; then
    if kubectl get pods -l app=mongodb -n neurashield &>/dev/null; then
      POD=$(kubectl get pods -l app=mongodb -n neurashield -o jsonpath="{.items[0].metadata.name}")
      
      # Copy backup to pod
      kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/mongodb_backup" -n neurashield $POD:/tmp/mongodb_backup
      
      # Restore database
      kubectl exec -n neurashield $POD -- mongorestore --drop /tmp/mongodb_backup
      
      # Cleanup
      kubectl exec -n neurashield $POD -- rm -rf /tmp/mongodb_backup
    else
      echo -e "${RED}MongoDB pod not found in Kubernetes, skipping database restore${NC}"
      return 1
    fi
  else
    echo -e "${RED}Database backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Database restore completed!${NC}"
}

# Restore function for Kubernetes AI models
k8s_restore_models() {
  echo -e "${YELLOW}Restoring AI models in Kubernetes environment...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models" ]; then
    if kubectl get pvc ai-models-pvc -n neurashield &>/dev/null; then
      # Create temp pod to access PVC
      echo "Creating temporary pod to access AI models PVC..."
      cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: restore-models-pod
  namespace: neurashield
spec:
  containers:
  - name: restore-container
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: models-volume
      mountPath: /models
  volumes:
  - name: models-volume
    persistentVolumeClaim:
      claimName: ai-models-pvc
EOF
      
      # Wait for pod to be ready
      kubectl wait --for=condition=Ready pod/restore-models-pod -n neurashield --timeout=60s
      
      # Clear existing data and copy the backup
      kubectl exec -n neurashield restore-models-pod -- sh -c "rm -rf /models/*"
      kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/models" -n neurashield restore-models-pod:/
      kubectl exec -n neurashield restore-models-pod -- sh -c "cp -r /models/* /models/ && chmod -R 755 /models/"
      
      # Clean up
      kubectl delete pod restore-models-pod -n neurashield
    else
      echo -e "${RED}AI models PVC not found in Kubernetes, skipping models restore${NC}"
      return 1
    fi
  else
    echo -e "${RED}AI models backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}AI models restore completed!${NC}"
}

# Restore function for Kubernetes blockchain data
k8s_restore_blockchain() {
  echo -e "${YELLOW}Restoring blockchain data in Kubernetes environment...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/blockchain" ]; then
    if kubectl get pvc blockchain-pvc -n neurashield &>/dev/null; then
      # Create temp pod to access PVC
      echo "Creating temporary pod to access blockchain PVC..."
      cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: restore-blockchain-pod
  namespace: neurashield
spec:
  containers:
  - name: restore-container
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: blockchain-volume
      mountPath: /blockchain
  volumes:
  - name: blockchain-volume
    persistentVolumeClaim:
      claimName: blockchain-pvc
EOF
      
      # Wait for pod to be ready
      kubectl wait --for=condition=Ready pod/restore-blockchain-pod -n neurashield --timeout=60s
      
      # Clear existing data and copy the backup
      kubectl exec -n neurashield restore-blockchain-pod -- sh -c "rm -rf /blockchain/*"
      kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/blockchain" -n neurashield restore-blockchain-pod:/
      kubectl exec -n neurashield restore-blockchain-pod -- sh -c "cp -r /blockchain/* /blockchain/ && chmod -R 755 /blockchain/"
      
      # Clean up
      kubectl delete pod restore-blockchain-pod -n neurashield
    else
      echo -e "${RED}Blockchain PVC not found in Kubernetes, skipping blockchain restore${NC}"
      return 1
    fi
  else
    echo -e "${RED}Blockchain backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Blockchain data restore completed!${NC}"
}

# Restore function for Kubernetes IPFS data
k8s_restore_ipfs() {
  echo -e "${YELLOW}Restoring IPFS data in Kubernetes environment...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/ipfs" ]; then
    if kubectl get pvc ipfs-pvc -n neurashield &>/dev/null; then
      # Create temp pod to access PVC
      echo "Creating temporary pod to access IPFS PVC..."
      cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: restore-ipfs-pod
  namespace: neurashield
spec:
  containers:
  - name: restore-container
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: ipfs-volume
      mountPath: /ipfs
  volumes:
  - name: ipfs-volume
    persistentVolumeClaim:
      claimName: ipfs-pvc
EOF
      
      # Wait for pod to be ready
      kubectl wait --for=condition=Ready pod/restore-ipfs-pod -n neurashield --timeout=60s
      
      # Clear existing data and copy the backup
      kubectl exec -n neurashield restore-ipfs-pod -- sh -c "rm -rf /ipfs/*"
      kubectl cp "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/ipfs" -n neurashield restore-ipfs-pod:/
      kubectl exec -n neurashield restore-ipfs-pod -- sh -c "cp -r /ipfs/* /ipfs/ && chmod -R 755 /ipfs/"
      
      # Clean up
      kubectl delete pod restore-ipfs-pod -n neurashield
    else
      echo -e "${RED}IPFS PVC not found in Kubernetes, skipping IPFS restore${NC}"
      return 1
    fi
  else
    echo -e "${RED}IPFS backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}IPFS data restore completed!${NC}"
}

# Restore function for Kubernetes configuration files
k8s_restore_configs() {
  echo -e "${YELLOW}Restoring configuration files in Kubernetes environment...${NC}"
  
  if [ -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs" ]; then
    # Backup current configs before applying restored ones
    echo "Backing up current configs before restore..."
    mkdir -p "/tmp/pre_restore_backup"
    kubectl get configmaps -n neurashield -o yaml > "/tmp/pre_restore_backup/configmaps.yaml"
    kubectl get secrets -n neurashield -o yaml > "/tmp/pre_restore_backup/secrets.yaml"
    
    # Apply backed up configs
    if [ -f "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/configmaps.yaml" ]; then
      echo "Restoring ConfigMaps..."
      kubectl apply -f "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/configmaps.yaml"
    fi
    
    if [ -f "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/secrets.yaml" ]; then
      echo "Restoring Secrets..."
      kubectl apply -f "${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/secrets.yaml"
    fi
    
    # Note: we're not automatically applying deployments/services as that could disrupt the system
    echo "Note: Deployments and Services configurations were not automatically applied."
    echo "      Review them manually at ${BACKUP_DIR}/${RESTORE_TIMESTAMP}/configs/"
  else
    echo -e "${RED}Configuration backup not found!${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Configuration restore completed!${NC}"
}

# Perform selected action
if [[ "$ACTION" == "backup" ]]; then
  echo -e "${YELLOW}Starting backup process with timestamp: ${TIMESTAMP}${NC}"
  mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"
  
  for component in "${BACKUP_COMPONENTS[@]}"; do
    if [[ "$ENVIRONMENT" == "docker" ]]; then
      case "$component" in
        "database")
          docker_backup_database
          ;;
        "models")
          docker_backup_models
          ;;
        "blockchain")
          docker_backup_blockchain
          ;;
        "ipfs")
          docker_backup_ipfs
          ;;
        "configs")
          docker_backup_configs
          ;;
        *)
          echo -e "${RED}Unknown component: $component${NC}"
          ;;
      esac
    elif [[ "$ENVIRONMENT" == "kubernetes" ]]; then
      case "$component" in
        "database")
          k8s_backup_database
          ;;
        "models")
          k8s_backup_models
          ;;
        "blockchain")
          k8s_backup_blockchain
          ;;
        "ipfs")
          k8s_backup_ipfs
          ;;
        "configs")
          k8s_backup_configs
          ;;
        *)
          echo -e "${RED}Unknown component: $component${NC}"
          ;;
      esac
    fi
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
  
  # If running interactively, confirm with the user 
  if [ -t 0 ]; then
    echo -e "${RED}WARNING: This operation will overwrite current data with backup data.${NC}"
    echo -e "${RED}Components to restore: ${BACKUP_COMPONENTS[*]}${NC}"
    read -p "Are you sure you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Restore operation cancelled."
      exit 0
    fi
  fi
  
  for component in "${BACKUP_COMPONENTS[@]}"; do
    if [[ "$ENVIRONMENT" == "docker" ]]; then
      case "$component" in
        "database")
          docker_restore_database
          ;;
        "models")
          docker_restore_models
          ;;
        "blockchain")
          docker_restore_blockchain
          ;;
        "ipfs")
          docker_restore_ipfs
          ;;
        "configs")
          docker_restore_configs
          ;;
        *)
          echo -e "${RED}Unknown component: $component${NC}"
          ;;
      esac
    elif [[ "$ENVIRONMENT" == "kubernetes" ]]; then
      case "$component" in
        "database")
          k8s_restore_database
          ;;
        "models")
          k8s_restore_models
          ;;
        "blockchain")
          k8s_restore_blockchain
          ;;
        "ipfs")
          k8s_restore_ipfs
          ;;
        "configs")
          k8s_restore_configs
          ;;
        *)
          echo -e "${RED}Unknown component: $component${NC}"
          ;;
      esac
    fi
  done
  
  echo -e "${GREEN}Restore process completed successfully!${NC}"
  
elif [[ "$ACTION" == "verify" ]]; then
  echo -e "${YELLOW}Starting backup verification for timestamp: ${RESTORE_TIMESTAMP}${NC}"
  
  # Check if backup exists
  if [[ ! -d "${BACKUP_DIR}/${RESTORE_TIMESTAMP}" ]]; then
    echo -e "${RED}Error: Backup with timestamp ${RESTORE_TIMESTAMP} not found!${NC}"
    exit 1
  fi
  
  VERIFICATION_FAILED=0
  
  for component in "${BACKUP_COMPONENTS[@]}"; do
    if ! verify_backup "$component"; then
      VERIFICATION_FAILED=1
    fi
  done
  
  if [[ "$VERIFICATION_FAILED" -eq 1 ]]; then
    echo -e "${RED}Backup verification failed!${NC}"
    exit 1
  else
    echo -e "${GREEN}All backup components verified successfully!${NC}"
  fi
fi 