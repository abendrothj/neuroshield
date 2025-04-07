/*
 * Test script to verify blockchain integration
 */

const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');

async function main() {
    try {
        // Load connection profile
        const ccpPath = path.resolve(__dirname, 'connection-profile.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new file system based wallet for managing identities
        const walletPath = path.join(__dirname, 'wallet');
        const wallet = await Wallets.newFileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check if admin identity exists in the wallet
        const identity = await wallet.get('admin');
        if (!identity) {
            console.log('Admin identity not found in the wallet');
            console.log('Run enroll-admin.js first');
            return;
        }

        // Create a new gateway for connecting to the peer node
        const gateway = new Gateway();
        await gateway.connect(ccp, {
            wallet,
            identity: 'admin',
            discovery: { enabled: true, asLocalhost: true }
        });

        // Get the network (channel) our contract is deployed to
        const network = await gateway.getNetwork('neurashield-channel');

        // Get the contract from the network
        const contract = network.getContract('neurashield');

        // Create a test event
        const eventId = `test-${Date.now()}`;
        const timestamp = new Date().toISOString();
        const eventType = 'ThreatDetection';
        const details = JSON.stringify({
            source: '192.168.1.1',
            destination: '10.0.0.1',
            protocol: 'TCP',
            severity: 'high',
            description: 'Potential unauthorized access attempt'
        });
        const ipfsHash = 'QmTest'; // This would normally be a real IPFS hash

        console.log(`Submitting transaction: LogEvent with ID ${eventId}`);
        await contract.submitTransaction('LogEvent', eventId, timestamp, eventType, details, ipfsHash);
        console.log('Transaction has been submitted');

        // Query all events to verify the new event was added
        console.log('Querying the ledger for all events...');
        const result = await contract.evaluateTransaction('QueryAllEvents');
        console.log(`Result: ${result.toString()}`);

        // Disconnect from the gateway
        await gateway.disconnect();

    } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}

main(); 