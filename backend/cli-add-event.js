/**
 * CLI script to add events directly to the blockchain
 * This uses the Fabric SDK with the same connection parameters as the CLI
 */

const fs = require('fs');
const path = require('path');
const { Gateway, Wallets } = require('fabric-network');

async function main() {
    try {
        // Set up the connection profile
        const ccpPath = path.resolve(__dirname, 'connection-profile.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new in-memory wallet
        const wallet = await Wallets.newInMemoryWallet();
        
        // Read the certificate and private key from files
        const certPath = '/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/signcerts/cert.pem';
        const keyPath = '/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore/c66fffa473e09eaadfd6a5ae27d98ac31bf18bd3d1f989d044dc2b63b3f73e34_sk';

        const cert = fs.readFileSync(certPath).toString();
        const key = fs.readFileSync(keyPath).toString();
        
        // Create a new identity in the wallet
        const identity = {
            credentials: {
                certificate: cert,
                privateKey: key,
            },
            mspId: 'Org1MSP',
            type: 'X.509',
        };
        await wallet.put('admin', identity);

        // Create a new gateway connection
        const gateway = new Gateway();
        await gateway.connect(ccp, {
            wallet,
            identity: 'admin',
            discovery: { enabled: false }
        });

        // Get the network and contract
        const network = await gateway.getNetwork('neurashield-channel');
        const contract = network.getContract('neurashield');

        // Create a unique ID for the event
        const eventId = `script-event-${Date.now()}`;
        const timestamp = new Date().toISOString();
        const eventType = 'SCRIPT-TEST';
        const details = 'Test event added from Node.js script';
        const ipfsHash = '';

        console.log('Submitting transaction...');
        
        // Submit the transaction to add an event
        await contract.submitTransaction(
            'LogEvent',
            eventId,
            timestamp,
            eventType,
            details,
            ipfsHash
        );

        console.log('Transaction submitted successfully');

        // Query all events to verify
        const result = await contract.evaluateTransaction('QueryAllEvents');
        const events = JSON.parse(result.toString());
        console.log('All events:');
        console.log(JSON.stringify(events, null, 2));

        // Disconnect from the gateway
        gateway.disconnect();

    } catch (error) {
        console.error(`Failed to add event: ${error}`);
        console.error(error.stack);
        process.exit(1);
    }
}

main(); 