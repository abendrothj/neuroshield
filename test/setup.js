/**
 * Test setup file for NeuraShield
 * 
 * This file is loaded before tests are run and sets up the test environment
 */

// Set environment variables for testing
process.env.NODE_ENV = 'test';
process.env.GCP_PROJECT_ID = 'test-project-id';
process.env.SERVICE_NAME = 'neurashield-test';
process.env.SERVICE_VERSION = '1.0.0-test';

// Set global timeout for tests
jest.setTimeout(30000);

// Custom test utilities
global.waitFor = async (callback, options = { timeout: 5000, interval: 100 }) => {
  const { timeout, interval } = options;
  const maxAttempts = timeout / interval;
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    try {
      return await callback();
    } catch (error) {
      attempts++;
      if (attempts >= maxAttempts) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  }
}; 