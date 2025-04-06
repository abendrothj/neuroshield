#!/bin/bash

# NeuraShield Backup Alert Notification Script
# This script sends notifications when backup or verification processes fail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default variables
ALERT_TYPE="backup_failed"
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
WEBHOOK_URL=${WEBHOOK_URL:-""}
EMAIL_RECIPIENT=${EMAIL_RECIPIENT:-"admin@example.com"}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}
DISCORD_WEBHOOK=${DISCORD_WEBHOOK:-""}
ENVIRONMENT="production"
DETAILS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --alert-type)
      ALERT_TYPE="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --webhook-url)
      WEBHOOK_URL="$2"
      shift 2
      ;;
    --email)
      EMAIL_RECIPIENT="$2"
      shift 2
      ;;
    --slack-webhook)
      SLACK_WEBHOOK="$2"
      shift 2
      ;;
    --discord-webhook)
      DISCORD_WEBHOOK="$2"
      shift 2
      ;;
    --environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --details)
      DETAILS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: backup-alert.sh [options]"
      echo "Options:"
      echo "  --alert-type TYPE       Alert type (backup_failed, verify_failed, etc.)"
      echo "  --timestamp TIMESTAMP   Timestamp of the event"
      echo "  --webhook-url URL       Generic webhook URL"
      echo "  --email EMAIL           Email recipient" 
      echo "  --slack-webhook URL     Slack webhook URL"
      echo "  --discord-webhook URL   Discord webhook URL"
      echo "  --environment ENV       Environment (production, staging, etc.)"
      echo "  --details TEXT          Additional details about the alert"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Get hostname and system information
HOST=$(hostname)
SYSTEM_INFO=$(uname -a)
IP_ADDRESS=$(hostname -I 2>/dev/null || echo "Unknown")

# Format alert message based on type
case "$ALERT_TYPE" in
  "backup_failed")
    SUBJECT="⚠️ [ALERT] NeuraShield Backup Failed - $ENVIRONMENT"
    MESSAGE="Backup process failed at $TIMESTAMP on $HOST ($IP_ADDRESS).\n\nSystem: $SYSTEM_INFO\n\nDetails: $DETAILS"
    ;;
  "verify_failed")
    SUBJECT="⚠️ [ALERT] NeuraShield Backup Verification Failed - $ENVIRONMENT"
    MESSAGE="Backup verification failed at $TIMESTAMP on $HOST ($IP_ADDRESS).\n\nSystem: $SYSTEM_INFO\n\nDetails: $DETAILS"
    ;;
  "restore_failed")
    SUBJECT="⚠️ [ALERT] NeuraShield Restore Failed - $ENVIRONMENT"
    MESSAGE="Restore process failed at $TIMESTAMP on $HOST ($IP_ADDRESS).\n\nSystem: $SYSTEM_INFO\n\nDetails: $DETAILS"
    ;;
  "backup_success")
    SUBJECT="✅ NeuraShield Backup Completed - $ENVIRONMENT"
    MESSAGE="Backup process completed successfully at $TIMESTAMP on $HOST ($IP_ADDRESS)."
    ;;
  *)
    SUBJECT="[INFO] NeuraShield Backup System - $ENVIRONMENT"
    MESSAGE="Event: $ALERT_TYPE\nTimestamp: $TIMESTAMP\nHost: $HOST ($IP_ADDRESS)\nDetails: $DETAILS"
    ;;
esac

# Send email notification
send_email() {
  if [[ -n "$EMAIL_RECIPIENT" ]]; then
    echo -e "${YELLOW}Sending email notification to $EMAIL_RECIPIENT...${NC}"
    if command -v mail &> /dev/null; then
      echo -e "$MESSAGE" | mail -s "$SUBJECT" "$EMAIL_RECIPIENT"
      echo -e "${GREEN}Email notification sent!${NC}"
    else
      echo -e "${RED}Error: mail command not found. Email notification failed.${NC}"
      return 1
    fi
  fi
}

# Send Slack notification
send_slack() {
  if [[ -n "$SLACK_WEBHOOK" ]]; then
    echo -e "${YELLOW}Sending Slack notification...${NC}"
    
    # Format message for Slack
    SLACK_COLOR="#FF0000"
    if [[ "$ALERT_TYPE" == *"success"* ]]; then
      SLACK_COLOR="#36a64f"
    fi
    
    SLACK_PAYLOAD=$(cat <<EOF
{
  "attachments": [
    {
      "fallback": "$SUBJECT",
      "color": "$SLACK_COLOR",
      "title": "$SUBJECT",
      "text": "$MESSAGE",
      "fields": [
        {
          "title": "Environment",
          "value": "$ENVIRONMENT",
          "short": true
        },
        {
          "title": "Timestamp",
          "value": "$TIMESTAMP",
          "short": true
        }
      ],
      "footer": "NeuraShield Backup System"
    }
  ]
}
EOF
)
    
    # Send to Slack
    curl -s -X POST -H "Content-type: application/json" --data "$SLACK_PAYLOAD" "$SLACK_WEBHOOK" > /dev/null
    if [[ $? -eq 0 ]]; then
      echo -e "${GREEN}Slack notification sent!${NC}"
    else
      echo -e "${RED}Error: Failed to send Slack notification.${NC}"
      return 1
    fi
  fi
}

# Send Discord notification
send_discord() {
  if [[ -n "$DISCORD_WEBHOOK" ]]; then
    echo -e "${YELLOW}Sending Discord notification...${NC}"
    
    # Format message for Discord
    DISCORD_COLOR="16711680"  # Red
    if [[ "$ALERT_TYPE" == *"success"* ]]; then
      DISCORD_COLOR="3066993"  # Green
    fi
    
    DISCORD_PAYLOAD=$(cat <<EOF
{
  "embeds": [
    {
      "title": "$SUBJECT",
      "description": "$MESSAGE",
      "color": $DISCORD_COLOR,
      "fields": [
        {
          "name": "Environment",
          "value": "$ENVIRONMENT",
          "inline": true
        },
        {
          "name": "Timestamp",
          "value": "$TIMESTAMP",
          "inline": true
        }
      ],
      "footer": {
        "text": "NeuraShield Backup System"
      }
    }
  ]
}
EOF
)
    
    # Send to Discord
    curl -s -X POST -H "Content-type: application/json" --data "$DISCORD_PAYLOAD" "$DISCORD_WEBHOOK" > /dev/null
    if [[ $? -eq 0 ]]; then
      echo -e "${GREEN}Discord notification sent!${NC}"
    else
      echo -e "${RED}Error: Failed to send Discord notification.${NC}"
      return 1
    fi
  fi
}

# Send generic webhook notification
send_webhook() {
  if [[ -n "$WEBHOOK_URL" ]]; then
    echo -e "${YELLOW}Sending webhook notification...${NC}"
    
    # Format message for generic webhook
    WEBHOOK_PAYLOAD=$(cat <<EOF
{
  "alert_type": "$ALERT_TYPE",
  "subject": "$SUBJECT",
  "message": "$MESSAGE",
  "timestamp": "$TIMESTAMP",
  "environment": "$ENVIRONMENT",
  "host": "$HOST",
  "ip_address": "$IP_ADDRESS",
  "details": "$DETAILS"
}
EOF
)
    
    # Send to webhook
    curl -s -X POST -H "Content-type: application/json" --data "$WEBHOOK_PAYLOAD" "$WEBHOOK_URL" > /dev/null
    if [[ $? -eq 0 ]]; then
      echo -e "${GREEN}Webhook notification sent!${NC}"
    else
      echo -e "${RED}Error: Failed to send webhook notification.${NC}"
      return 1
    fi
  fi
}

# Send all configured notifications
send_email
send_slack
send_discord
send_webhook

echo -e "${GREEN}Alert notifications completed!${NC}" 