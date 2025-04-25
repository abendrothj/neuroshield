#!/usr/bin/env node
/**
 * Command-line tool for managing blockchain identities
 * 
 * Usage:
 *   node manage-identities.js [command]
 * 
 * Commands:
 *   init                - Initialize all required identities
 *   enroll-admin        - Enroll the admin user
 *   register <userId>   - Register and enroll a new user
 *   service <serviceId> - Create a service identity
 *   list                - List all identities in the wallet
 *   verify <identity>   - Verify an identity works with the network
 *   delete <identity>   - Delete an identity from the wallet
 */

require('dotenv').config();
const identityManager = require('./identity-manager');

// Process command line arguments
const args = process.argv.slice(2);
const command = args[0];

async function main() {
  try {
    // Create logs directory if it doesn't exist
    const fs = require('fs');
    const path = require('path');
    const logsDir = path.join(__dirname, 'logs');
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    
    switch (command) {
      case 'init':
        console.log('Initializing all required identities...');
        const success = await identityManager.initializeIdentities();
        if (success) {
          console.log('✅ All identities initialized successfully');
        } else {
          console.error('❌ Failed to initialize identities');
          process.exit(1);
        }
        break;
        
      case 'enroll-admin':
        console.log('Enrolling admin user...');
        try {
          // Try MSP enrollment first
          await identityManager.enrollAdminFromMSP();
          console.log('✅ Admin enrolled successfully using MSP materials');
        } catch (mspError) {
          console.log(`MSP enrollment failed: ${mspError.message}`);
          console.log('Trying CA enrollment...');
          
          try {
            await identityManager.enrollAdminWithCA();
            console.log('✅ Admin enrolled successfully using CA');
          } catch (caError) {
            console.error(`❌ CA enrollment also failed: ${caError.message}`);
            process.exit(1);
          }
        }
        break;
        
      case 'register':
        const userId = args[1];
        if (!userId) {
          console.error('❌ User ID is required');
          console.log('Usage: node manage-identities.js register <userId>');
          process.exit(1);
        }
        
        console.log(`Registering user ${userId}...`);
        await identityManager.registerUser(userId);
        console.log(`✅ User ${userId} registered successfully`);
        break;
        
      case 'service':
        const serviceId = args[1];
        if (!serviceId) {
          console.error('❌ Service ID is required');
          console.log('Usage: node manage-identities.js service <serviceId>');
          process.exit(1);
        }
        
        console.log(`Creating service identity ${serviceId}...`);
        await identityManager.createServiceIdentity(serviceId);
        console.log(`✅ Service identity ${serviceId} created successfully`);
        break;
        
      case 'list':
        console.log('Listing all identities in the wallet...');
        const identities = await identityManager.listIdentities();
        
        if (identities.length === 0) {
          console.log('No identities found in the wallet');
        } else {
          console.log('Identities:');
          identities.forEach((id, index) => {
            console.log(`  ${index + 1}. ${id}`);
          });
        }
        break;
        
      case 'verify':
        const verifyId = args[1];
        if (!verifyId) {
          console.error('❌ Identity label is required');
          console.log('Usage: node manage-identities.js verify <identity>');
          process.exit(1);
        }
        
        console.log(`Verifying identity ${verifyId}...`);
        const isValid = await identityManager.verifyIdentity(verifyId);
        
        if (isValid) {
          console.log(`✅ Identity ${verifyId} is valid and working correctly`);
        } else {
          console.error(`❌ Identity ${verifyId} is not valid or not working`);
          process.exit(1);
        }
        break;
        
      case 'delete':
        const deleteId = args[1];
        if (!deleteId) {
          console.error('❌ Identity label is required');
          console.log('Usage: node manage-identities.js delete <identity>');
          process.exit(1);
        }
        
        console.log(`Deleting identity ${deleteId}...`);
        const deleted = await identityManager.deleteIdentity(deleteId);
        
        if (deleted) {
          console.log(`✅ Identity ${deleteId} deleted successfully`);
        } else {
          console.error(`❌ Failed to delete identity ${deleteId}`);
          process.exit(1);
        }
        break;
        
      default:
        console.log(`
NeuraShield Identity Management CLI

Usage:
  node manage-identities.js [command]

Commands:
  init                - Initialize all required identities
  enroll-admin        - Enroll the admin user
  register <userId>   - Register and enroll a new user
  service <serviceId> - Create a service identity
  list                - List all identities in the wallet
  verify <identity>   - Verify an identity works with the network
  delete <identity>   - Delete an identity from the wallet
        `);
        break;
    }
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main(); 