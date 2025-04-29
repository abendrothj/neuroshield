# Deployment Scripts

This directory contains scripts related to deploying and configuring the NeuraShield application.

## Scripts

- `production-setup.sh` - Sets up the NeuraShield production environment
- `new-env-settings.sh` - Updates environment settings for different deployment scenarios

## Usage

These scripts should be run from the project root directory:

```bash
# Example: Set up a production environment
bash scripts/deployment/production-setup.sh

# Example: Update environment settings
bash scripts/deployment/new-env-settings.sh
```

## Notes

- The production setup script configures all components for a production environment
- Use these scripts with caution in existing environments as they may modify configurations 