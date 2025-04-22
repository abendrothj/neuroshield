# Blockchain Deployment Guide for NeuraShield

This guide provides instructions for deploying and managing the Hyperledger Fabric blockchain component of NeuraShield in both development and production environments.

## Overview

NeuraShield uses Hyperledger Fabric to provide an immutable, secure ledger for recording security events detected by the AI model. This blockchain implementation ensures:

- Tamper-proof logging of security events
- Cryptographic verification of event data
- Decentralized storage of critical security information
- Auditability and compliance with industry regulations

## Architecture

The blockchain component consists of:

1. **Fabric Network**: A Hyperledger Fabric network with a single organization (Org1)
2. **Chaincode**: Go-based smart contract for logging security events
3. **Integration**: Backend service that connects to the Fabric network via the Fabric SDK for Node.js
4. **IPFS Integration**: Optional storage of large log files on IPFS with references stored on the blockchain

## Development Environment Setup

For local development, we use Docker Compose to run a simplified Fabric network:

1. **Set up the network**:
   ```bash
   cd /home/jub/Cursor/neurashield/fabric-setup
   ./bootstrap.sh
   ```

2. **Start the Fabric network**:
   ```bash
   cd /home/jub/Cursor/neurashield/blockchain/network
   docker-compose -f docker-compose-fabric.yml up -d
   ```

3. **Install and instantiate the chaincode**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./deploy-chaincode.sh
   ```

4. **Test the blockchain connection**:
   ```bash
   cd /home/jub/Cursor/neurashield/backend
   node test-blockchain.js
   ```

## Production Deployment

For production environments, we use Kubernetes to manage the Fabric network:

1. **Prepare Kubernetes cluster**:
   Ensure your Kubernetes cluster is properly set up and configured.

2. **Create crypto materials**:
   For production, generate proper certificates using Fabric CA or cryptogen.
   Store these securely in Kubernetes secrets.

3. **Deploy the blockchain network**:
   ```bash
   cd /home/jub/Cursor/neurashield/k8s
   kubectl apply -f blockchain-deployment.yaml
   kubectl apply -f fabric-configmaps.yaml
   ```

4. **Update connection profile**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./update-connection-profile.sh
   ```

5. **Deploy and initialize chaincode**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./update-chaincode-k8s.sh
   ```

## Updating Chaincode

When you need to update the chaincode (smart contract):

1. **For development**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./deploy-chaincode.sh
   ```

2. **For production**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./update-chaincode-k8s.sh --version 1.x  # Increment version number for each update
   ```

## Security Considerations

1. **MSP Credentials**: In production, properly secure your MSP (Membership Service Provider) credentials
2. **TLS**: Always enable TLS in production environments
3. **Access Control**: Use Fabric's built-in access control to restrict chaincode access
4. **Private Data**: For sensitive information, consider using Fabric's private data collections
5. **Monitoring**: Set up monitoring for your blockchain network to detect issues

## Troubleshooting

Common issues and solutions:

1. **Connection failures**:
   - Check if the Fabric network is running: `kubectl get pods -l app=hyperledger`
   - Verify network connectivity: `kubectl exec -it <peer-pod> -- ping fabric-orderer`
   - Check TLS certificate paths in connection profile

2. **Chaincode errors**:
   - Check chaincode logs: `kubectl logs <peer-pod> | grep chaincode`
   - Verify chaincode installation: `kubectl exec <cli-pod> -- peer lifecycle chaincode queryinstalled`

3. **Performance issues**:
   - Consider scaling the blockchain network (adding more peers)
   - Optimize chaincode (minimize state operations)
   - Use CouchDB instead of LevelDB for complex queries

## Backup and Recovery

1. **Regular backups**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./backup-recovery.sh --backup blockchain
   ```

2. **Recovery procedure**:
   ```bash
   cd /home/jub/Cursor/neurashield/scripts
   ./backup-recovery.sh --restore blockchain --timestamp <timestamp>
   ```

## Further Resources

- [Hyperledger Fabric Documentation](https://hyperledger-fabric.readthedocs.io/)
- [Fabric SDK for Node.js](https://hyperledger.github.io/fabric-sdk-node/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/) 