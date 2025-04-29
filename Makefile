# NeuraShield Makefile

# Directory definitions
SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts
BLOCKCHAIN_SCRIPTS := $(SCRIPTS_DIR)/blockchain
SERVER_SCRIPTS := $(SCRIPTS_DIR)/server
DEPLOYMENT_SCRIPTS := $(SCRIPTS_DIR)/deployment
UTILS_SCRIPTS := $(SCRIPTS_DIR)/utils
TESTING_SCRIPTS := $(SCRIPTS_DIR)/testing

# Help command
.PHONY: help
help:
	@echo "NeuraShield - Available commands:"
	@echo "  make start          - Start the NeuraShield platform"
	@echo "  make stop           - Stop the NeuraShield platform"
	@echo "  make setup-network  - Set up the Fabric network"
	@echo "  make deploy         - Deploy chaincode to the network"
	@echo "  make reset-admin    - Reset admin identity for blockchain access"
	@echo "  make test           - Run integration tests"
	@echo "  make test-simple    - Run simple verification tests"
	@echo "  make clean          - Clean up temporary files and logs"
	@echo "  make production     - Set up production environment"

# Start the NeuraShield platform
.PHONY: start
start:
	@bash $(PROJECT_ROOT)/start.sh

# Stop the NeuraShield platform
.PHONY: stop
stop:
	@bash $(PROJECT_ROOT)/stop.sh

# Set up the Fabric network
.PHONY: setup-network
setup-network:
	@bash $(BLOCKCHAIN_SCRIPTS)/setup-fabric-network.sh

# Deploy chaincode to the network
.PHONY: deploy
deploy:
	@bash $(BLOCKCHAIN_SCRIPTS)/deploy-neurashield.sh

# Reset admin identity
.PHONY: reset-admin
reset-admin:
	@bash $(SERVER_SCRIPTS)/reset-admin.sh

# Run integration tests
.PHONY: test
test:
	@bash $(TESTING_SCRIPTS)/test-integration.sh

# Run simple tests
.PHONY: test-simple
test-simple:
	@bash $(TESTING_SCRIPTS)/test-simple.sh

# Clean up temporary files
.PHONY: clean
clean:
	@bash $(UTILS_SCRIPTS)/cleanup.sh

# Set up production environment
.PHONY: production
production:
	@bash $(DEPLOYMENT_SCRIPTS)/production-setup.sh

# Additional targets for convenience

# Run server only
.PHONY: server
server:
	@bash $(SERVER_SCRIPTS)/run-server.sh

# Update environment settings
.PHONY: update-env
update-env:
	@bash $(DEPLOYMENT_SCRIPTS)/new-env-settings.sh 