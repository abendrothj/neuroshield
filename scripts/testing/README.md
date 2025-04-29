# Testing Scripts

This directory contains scripts related to testing the NeuraShield system.

## Scripts

- `test-integration.sh` - Runs integration tests across all components
- `test-simple.sh` - Runs simple tests for quick verification

## Usage

These scripts should be run from the project root directory:

```bash
# Example: Run integration tests
bash scripts/testing/test-integration.sh

# Example: Run simple tests
bash scripts/testing/test-simple.sh
```

## Notes

- Integration tests require all components to be properly set up and running
- The simple tests are useful for quickly verifying specific functionality
- Check test outputs in the `/logs` directory 