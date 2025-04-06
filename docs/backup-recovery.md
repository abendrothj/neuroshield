# NeuraShield Backup and Recovery System

This document provides comprehensive information on the backup and recovery system implemented for the NeuraShield platform. This system is designed to ensure data safety and business continuity across both Kubernetes and Docker Compose environments.

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Backup Schedule](#backup-schedule)
4. [Backup Storage](#backup-storage)
5. [Backup Verification](#backup-verification)
6. [Restore Process](#restore-process)
7. [Alert Notifications](#alert-notifications)
8. [Manual Operations](#manual-operations)
9. [Troubleshooting](#troubleshooting)

## Overview

The NeuraShield backup and recovery system performs regular automated backups of critical system components, verifies the integrity of these backups, and provides a straightforward process for data restoration in case of emergencies.

The system is designed to work across both deployment environments:
- Kubernetes deployments
- Docker Compose deployments

## Components

The following components are included in the backup process:

| Component | Description | Data Location |
|-----------|-------------|---------------|
| Database | MongoDB data | `/app/backend/data/db` (Docker) or MongoDB pods (K8s) |
| AI Models | Trained ML models | `/app/ai_models/models` or `ai-models-pvc` volume |
| Blockchain | Blockchain data | `/app/backend/data/blockchain` or `blockchain-pvc` volume |
| IPFS | Distributed file storage | `/data/ipfs` or `ipfs-pvc` volume |
| Configs | System configuration | Environment variables, ConfigMaps, Secrets |

## Backup Schedule

Backups are performed according to the following schedule:

- **Full Backup**: Daily at 1:00 AM
- **Backup Verification**: Daily at 5:00 AM (after backup completion)
- **Retention Period**: 30 days (configurable)

## Backup Storage

Backup data is stored in the following locations:

- **Kubernetes**: In a dedicated `backup-pvc` Persistent Volume Claim
- **Docker Compose**: In the `/opt/neurashield/backups` directory on the host

Each backup is stored in a timestamped directory (`YYYYMMDD_HHMMSS`) containing subdirectories for each component.

## Backup Verification

The system includes an automated verification process that checks:

1. Existence of backup data for each component
2. Basic integrity of the backup files
3. Readability of configuration files

Verification failures trigger alert notifications to system administrators.

## Restore Process

### Automated Restore

The system provides both a command-line interface and a web interface for restore operations:

#### Command Line Restore

```bash
# For Kubernetes environment
kubectl exec -it deploy/backup-service -- /scripts/backup-recovery.sh --restore --timestamp 20230101_120000 --components database,models

# For Docker Compose environment
docker-compose exec backup /scripts/backup-recovery.sh --restore --timestamp 20230101_120000 --components database,models
```

#### Web Interface

1. Access the restore service: `https://[your-domain]/restore/` (Kubernetes) or `http://localhost:8080` (Docker)
2. Select the backup timestamp from the dropdown
3. Choose components to restore
4. Confirm the restore operation

### Restore Precautions

Before performing a restore:

1. All services accessing the affected components should be stopped
2. A backup of the current state should be created (happens automatically)
3. Proper access permissions should be in place

## Alert Notifications

The system can send alerts through multiple channels:

- Email notifications
- Slack webhooks
- Discord webhooks
- Custom webhook endpoints

Alerts are sent for:
- Backup failures
- Verification failures
- Restore operations (success and failure)

Configuration for alerts is stored in environment variables or as Kubernetes secrets.

## Manual Operations

### Manual Backup Trigger

```bash
# For Kubernetes
kubectl exec -it deploy/backup-service -- /scripts/backup-recovery.sh --backup

# For Docker Compose
docker-compose exec backup /scripts/backup-recovery.sh --backup
```

### Manual Verification

```bash
# Verify latest backup
/scripts/backup-recovery.sh --verify

# Verify specific backup
/scripts/backup-recovery.sh --verify --timestamp 20230101_120000
```

### Manual Restore

```bash
# Full system restore
/scripts/backup-recovery.sh --restore --timestamp 20230101_120000

# Partial restore (specific components)
/scripts/backup-recovery.sh --restore --timestamp 20230101_120000 --components database,configs
```

## Troubleshooting

### Common Issues

1. **Backup Failure**
   - Check disk space availability
   - Verify service account permissions
   - Check component availability

2. **Verification Failure**
   - Check backup logs for specific component errors
   - Ensure backup storage is accessible

3. **Restore Failure**
   - Verify backup timestamp exists
   - Check component status and readiness
   - Ensure sufficient permissions for restore operations

### Log Locations

- **Kubernetes**: Use `kubectl logs -f [pod-name]` for the backup/restore pods
- **Docker Compose**: Use `docker-compose logs backup` or check the backup service's logs 