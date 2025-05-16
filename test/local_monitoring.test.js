/**
 * NeuraShield Local Monitoring Tests
 */

// Import mocks
jest.mock('../backend/src/local-observability', () => require('./mocks/local-observability.mock'));

const express = require('express');
const request = require('supertest');
const { initObservability } = require('../backend/src/local-observability');

describe('Local Monitoring', () => {
  let observability;
  let app;

  beforeEach(() => {
    // Initialize observability
    observability = initObservability();
    
    // Setup Express app with monitoring middleware
    app = express();
    
    // Add metrics middleware
    app.use(observability.middleware.metrics);
    
    // Add error handler
    app.use(observability.middleware.errorReporting);
    
    // Add test routes
    app.get('/api/test', (req, res) => {
      req.log('info', 'Test request received', { testId: 123 });
      res.status(200).json({ success: true });
    });
    
    app.get('/api/test-error', (req, res, next) => {
      next(new Error('Test error for error reporting'));
    });
    
    app.get('/api/blockchain/test', (req, res) => {
      // Simulate blockchain operation with monitoring
      observability.metricsTracker.blockchainTransactionsCounter.add(1, {
        operation: 'test',
        status: 'success'
      });
      res.status(200).json({ success: true });
    });
    
    app.get('/api/ai/test', (req, res) => {
      // Simulate AI operation with monitoring
      observability.metricsTracker.aiPredictionsCounter.add(1, {
        model: 'test-model',
        status: 'success'
      });
      res.status(200).json({ success: true });
    });
    
    // Add error handler - this should be last
    app.use((err, req, res, next) => {
      // Make sure our error reporting middleware gets called
      observability.errorHandler.report(err, req);
      res.status(500).json({ error: err.message });
    });
  });

  test('should initialize observability components', () => {
    expect(observability).toBeDefined();
    expect(observability.logger).toBeDefined();
    expect(observability.metricsTracker).toBeDefined();
    expect(observability.errorHandler).toBeDefined();
    expect(observability.middleware).toBeDefined();
    expect(observability.middleware.metrics).toBeDefined();
    expect(observability.middleware.errorReporting).toBeDefined();
  });

  test('should record API requests', async () => {
    // Make a request to the test route
    await request(app).get('/api/test').expect(200);
    
    // Check that the API request was recorded
    const counter = observability.metricsTracker.apiRequestsCounter;
    expect(counter.value).toBe(1);
    expect(counter.labels[0].method).toBe('GET');
    expect(counter.labels[0].path).toBe('/api/test');
  });

  test('should record blockchain transactions', async () => {
    // Make a request to the blockchain test route
    await request(app).get('/api/blockchain/test').expect(200);
    
    // Check that the blockchain transaction was recorded
    const counter = observability.metricsTracker.blockchainTransactionsCounter;
    expect(counter.value).toBe(1);
    expect(counter.labels[0].operation).toBe('test');
    expect(counter.labels[0].status).toBe('success');
  });

  test('should record AI predictions', async () => {
    // Make a request to the AI test route
    await request(app).get('/api/ai/test').expect(200);
    
    // Check that the AI prediction was recorded
    const counter = observability.metricsTracker.aiPredictionsCounter;
    expect(counter.value).toBe(1);
    expect(counter.labels[0].model).toBe('test-model');
    expect(counter.labels[0].status).toBe('success');
  });

  test('should report errors', async () => {
    // Make a request that will trigger an error
    await request(app).get('/api/test-error').expect(500);
    
    // Check that the error was reported
    expect(observability.errorHandler.errors.length).toBe(1);
    expect(observability.errorHandler.errors[0].error.message).toBe('Test error for error reporting');
  });
  
  test('should record API latency', async () => {
    // Make a request to the test route
    await request(app).get('/api/test').expect(200);
    
    // Check that the API latency was recorded
    const histogram = observability.metricsTracker.apiLatencyHistogram;
    expect(histogram.records.length).toBe(1);
    expect(histogram.records[0].labels.method).toBe('GET');
    expect(histogram.records[0].labels.path).toBe('/api/test');
    expect(histogram.records[0].labels.status).toBe(200);
    expect(typeof histogram.records[0].value).toBe('number');
  });
  
  test('should log requests', async () => {
    // Make a request to the test route
    await request(app).get('/api/test').expect(200);
    
    // Verify that logger was called
    expect(observability.logger.info).toHaveBeenCalled();
  });
}); 