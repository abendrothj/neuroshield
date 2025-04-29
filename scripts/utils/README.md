# Utility Scripts

This directory contains utility scripts for the NeuraShield platform.

## Scripts

- `cleanup.sh` - Cleans up temporary files, logs, and optionally Docker cache

## Usage

These scripts should be run from the project root directory:

```bash
# Example: Clean up temporary files and logs
bash scripts/utils/cleanup.sh
```

## Notes

- The cleanup script safely removes temporary files while preserving directory structures
- Log files larger than 10MB will be rotated
- Docker cache cleanup is optional and requires user confirmation 