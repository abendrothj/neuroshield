/*
 * Enrolls an admin user to the wallet
 */

const fs = require('fs');
const path = require('path');
const FabricCAServices = require('fabric-ca-client');
const { Wallets } = require('fabric-network');

async function main() {
  try {
    // Create a new file system based wallet for managing identities.
    const walletPath = path.join(__dirname, 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    console.log(`Wallet path: ${walletPath}`);

    // Check if admin identity exists
    const identity = await wallet.get('admin');
    if (identity) {
      console.log('An identity for the admin user "admin" already exists in the wallet');
      return;
    }

    // Copy the MSP materials from the test-network to the wallet
    const mspPath = '/home/jub/Cursor/neurashield/fabric-setup/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp';
    
    if (!fs.existsSync(mspPath)) {
      console.error(`MSP path not found: ${mspPath}`);
      return;
    }

    // Read the certificate and private key
    const certPath = path.join(mspPath, 'signcerts', 'cert.pem');
    const keyDir = path.join(mspPath, 'keystore');
    console.log(`Looking for key files in: ${keyDir}`);
    const keyFiles = fs.readdirSync(keyDir);
    console.log(`Found keystore files: ${keyFiles.join(', ')}`);
    const keyPath = path.join(keyDir, keyFiles[0]);

    const cert = fs.readFileSync(certPath).toString();
    const key = fs.readFileSync(keyPath).toString();

    // Load the MSP ID from the config.json
    const mspId = 'Org1MSP';

    // Store the identity in the wallet
    const x509Identity = {
      credentials: {
        certificate: cert,
        privateKey: key,
      },
      mspId: mspId,
      type: 'X.509',
    };

    await wallet.put('admin', x509Identity);
    console.log('Successfully enrolled admin user "admin" and imported it into the wallet');

  } catch (error) {
    console.error(`Failed to enroll admin user "admin": ${error}`);
    process.exit(1);
  }
}

main(); 