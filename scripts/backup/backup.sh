#!/bin/bash

# NeuraShield Backup and Recovery Script
# This script handles backup, recovery, and alerting for the NeuraShield system

set -e

# Configuration
BACKUP_DIR="backups"
LOG_DIR="logs"
ALERT_EMAIL="admin@neurashield.com"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Print with color
print() {
    echo -e "${GREEN}[BACKUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Send alert
send_alert() {
    local subject=$1
    local message=$2
    
    echo "$message" | mail -s "$subject" $ALERT_EMAIL
    print "Alert sent: $subject"
}

# Backup database
backup_database() {
    print "Backing up database..."
    
    # Create backup directory
    mkdir -p $BACKUP_DIR/database/$TIMESTAMP
    
    # Backup PostgreSQL
    pg_dump -U postgres neurashield > $BACKUP_DIR/database/$TIMESTAMP/db_backup.sql
    
    # Compress backup
    gzip $BACKUP_DIR/database/$TIMESTAMP/db_backup.sql
    
    print "Database backup complete"
}

# Backup blockchain data
backup_blockchain() {
    print "Backing up blockchain data..."
    
    # Create backup directory
    mkdir -p $BACKUP_DIR/blockchain/$TIMESTAMP
    
    # Backup ledger data
    cp -r blockchain/ledger $BACKUP_DIR/blockchain/$TIMESTAMP/
    
    # Backup chaincode
    cp -r backend/chaincode $BACKUP_DIR/blockchain/$TIMESTAMP/
    
    print "Blockchain backup complete"
}

# Backup AI models
backup_models() {
    print "Backing up AI models..."
    
    # Create backup directory
    mkdir -p $BACKUP_DIR/models/$TIMESTAMP
    
    # Backup model files
    cp -r models/* $BACKUP_DIR/models/$TIMESTAMP/
    
    print "AI models backup complete"
}

# Cleanup old backups
cleanup_backups() {
    print "Cleaning up old backups..."
    
    find $BACKUP_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;
    
    print "Cleanup complete"
}

# Restore database
restore_database() {
    local backup_dir=$1
    
    print "Restoring database from $backup_dir..."
    
    # Stop services
    docker-compose down
    
    # Restore database
    gunzip -c $backup_dir/db_backup.sql.gz | psql -U postgres neurashield
    
    # Start services
    docker-compose up -d
    
    print "Database restore complete"
}

# Restore blockchain
restore_blockchain() {
    local backup_dir=$1
    
    print "Restoring blockchain from $backup_dir..."
    
    # Stop blockchain network
    cd blockchain
    ./network.sh down
    cd ..
    
    # Restore ledger data
    cp -r $backup_dir/ledger blockchain/
    
    # Restore chaincode
    cp -r $backup_dir/chaincode backend/
    
    # Start blockchain network
    cd blockchain
    ./network.sh up
    cd ..
    
    print "Blockchain restore complete"
}

# Restore models
restore_models() {
    local backup_dir=$1
    
    print "Restoring models from $backup_dir..."
    
    # Restore model files
    cp -r $backup_dir/* models/
    
    print "Models restore complete"
}

# Main backup function
backup() {
    print "Starting backup process..."
    
    # Create backup directory
    mkdir -p $BACKUP_DIR/$TIMESTAMP
    
    # Perform backups
    backup_database
    backup_blockchain
    backup_models
    
    # Cleanup old backups
    cleanup_backups
    
    print "Backup process complete"
}

# Main restore function
restore() {
    local backup_dir=$1
    
    if [ -z "$backup_dir" ]; then
        print_error "Backup directory not specified"
        exit 1
    fi
    
    if [ ! -d "$backup_dir" ]; then
        print_error "Backup directory does not exist"
        exit 1
    fi
    
    print "Starting restore process from $backup_dir..."
    
    # Perform restores
    restore_database $backup_dir
    restore_blockchain $backup_dir
    restore_models $backup_dir
    
    print "Restore process complete"
}

# Main function
main() {
    case "$1" in
        backup)
            backup
            ;;
        restore)
            restore "$2"
            ;;
        *)
            print_error "Usage: $0 {backup|restore [backup_dir]}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 