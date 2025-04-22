#!/bin/bash

# Make all scripts executable
find scripts/ -name "*.sh" -exec chmod +x {} \;
find k8s/ -name "*.sh" -exec chmod +x {} \;

echo "All scripts made executable!" 