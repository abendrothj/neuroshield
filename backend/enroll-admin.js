/**
 * Enrollment script for NeuraShield admin user
 * 
 * This script enrolls the admin user for NeuraShield blockchain integration
 * using existing MSP materials from the running Fabric network.
 */

const identityManager = require('./identity-manager');
const path = require('path');
const fs = require('fs');
const { Wallets } = require('fabric-network');

// Path for the wallet directory
const walletPath = path.join(__dirname, 'wallet');

async function main() {
  try {
    console.log('Starting admin enrollment process...');
    
    // Create wallet directory if it doesn't exist
    if (!fs.existsSync(walletPath)) {
      fs.mkdirSync(walletPath, { recursive: true });
      console.log(`Created wallet directory at ${walletPath}`);
    }
    
    // Enroll admin user
    try {
      console.log('Trying to enroll admin from MSP materials...');
      await identityManager.enrollAdminFromMSP();
    } catch (mspError) {
      console.warn(`MSP enrollment failed: ${mspError.message}`);
      console.log('Trying CA enrollment as fallback...');
      
      try {
        await identityManager.enrollAdminWithCA();
      } catch (caError) {
        console.error(`CA enrollment also failed: ${caError.message}`);
        throw new Error('All enrollment methods failed');
      }
    }
    
    // Verify the admin identity was created
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    const adminIdentity = await wallet.get('admin');
    
    if (adminIdentity) {
      console.log('Admin enrollment successful!');
      console.log('You can now start the NeuraShield server');
    } else {
      console.error('Enrollment appeared to succeed but no identity was found in wallet');
      process.exit(1);
    }
    
  } catch (error) {
    console.error(`Failed to enroll admin user: ${error}`);
    process.exit(1);
  }
}

main(); 