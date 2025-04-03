const express = require('express');
const { Gateway, Wallets } = require('fabric-network');
const { create } = require('ipfs-http-client');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const app = express();
app.use(express.json());

// IPFS configuration
const ipfs = create({ url: process.env.IPFS_URL || 'http://localhost:5001' });

// Fabric configuration
const ccpPath = path.resolve(__dirname, '..', 'connection-profile.json');
const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

// Initialize Fabric gateway
async function getGateway() {
    const wallet = await Wallets.newFileSystemWallet(path.join(__dirname, 'wallet'));
    const gateway = new Gateway();
    await gateway.connect(ccp, { wallet, identity: 'admin', discovery: { enabled: true, asLocalhost: true } });
    return gateway;
}

// Routes
app.post('/api/events', async (req, res) => {
    try {
        const { id, timestamp, type, details } = req.body;
        
        // Upload details to IPFS
        const ipfsResult = await ipfs.add(JSON.stringify(details));
        const ipfsHash = ipfsResult.path;

        // Store event on Fabric
        const gateway = await getGateway();
        const network = await gateway.getNetwork('neurashield-channel');
        const contract = network.getContract('neurashield');

        await contract.submitTransaction('LogEvent', id, timestamp, type, details, ipfsHash);
        
        res.json({ success: true, message: 'Event logged successfully', ipfsHash });
    } catch (error) {
        console.error(`Failed to log event: ${error}`);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/events/:id', async (req, res) => {
    try {
        const gateway = await getGateway();
        const network = await gateway.getNetwork('neurashield-channel');
        const contract = network.getContract('neurashield');

        const result = await contract.evaluateTransaction('QueryEvent', req.params.id);
        res.json(JSON.parse(result.toString()));
    } catch (error) {
        console.error(`Failed to query event: ${error}`);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/events', async (req, res) => {
    try {
        const gateway = await getGateway();
        const network = await gateway.getNetwork('neurashield-channel');
        const contract = network.getContract('neurashield');

        const result = await contract.evaluateTransaction('QueryAllEvents');
        res.json(JSON.parse(result.toString()));
    } catch (error) {
        console.error(`Failed to query all events: ${error}`);
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
}); 