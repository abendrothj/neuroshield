/**
 * API Routes for NeuraShield
 */

const express = require('express');
const { body, param, validationResult } = require('express-validator');

// Input validation middleware
const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
};

module.exports = (observability) => {
  const router = express.Router();
  const { logger, metricsTracker } = observability;

  // Health check route
  router.get('/health', (req, res) => {
    res.status(200).json({ 
      status: 'healthy',
      version: '1.0.0',
      timestamp: new Date().toISOString()
    });
  });

  // Events routes
  router.post('/events', 
    [
      body('type').isString().withMessage('Type must be a string'),
      body('severity').isIn(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).withMessage('Severity must be one of: LOW, MEDIUM, HIGH, CRITICAL'),
      body('details').isObject().withMessage('Details must be an object'),
      validate
    ],
    async (req, res) => {
      try {
        const { type, severity, details } = req.body;
        
        // Log the event
        logger.info('Security event received', { type, severity });
        
        // Generate event ID
        const id = `event-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        // Record metrics
        metricsTracker.apiRequestsCounter.add(1, {
          method: 'POST',
          path: '/api/events',
          event_type: type,
          severity
        });
        
        // For now, just return successfully
        res.status(201).json({ 
          id,
          type,
          severity,
          timestamp: new Date().toISOString(),
          status: 'recorded'
        });
      } catch (error) {
        logger.error('Failed to process event', { error: error.message });
        res.status(500).json({ error: 'Failed to process event' });
      }
    }
  );

  // Get all events (stub)
  router.get('/events', async (req, res) => {
    try {
      // Return stub data for now
      res.json({
        events: [
          {
            id: 'event-1',
            type: 'THREAT_DETECTED',
            severity: 'HIGH',
            timestamp: new Date().toISOString(),
          },
          {
            id: 'event-2',
            type: 'ANOMALY_DETECTED',
            severity: 'MEDIUM',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
          }
        ]
      });
    } catch (error) {
      logger.error('Failed to retrieve events', { error: error.message });
      res.status(500).json({ error: 'Failed to retrieve events' });
    }
  });

  // Get single event by ID (stub)
  router.get('/events/:id', 
    [
      param('id').isString().withMessage('ID must be a string'),
      validate
    ],
    async (req, res) => {
      try {
        const { id } = req.params;
        
        // Stub response
        res.json({
          id,
          type: 'THREAT_DETECTED',
          severity: 'HIGH',
          timestamp: new Date().toISOString(),
          details: {
            source: '192.168.1.100',
            target: '10.0.0.1',
            description: 'Potential SQL injection attempt'
          }
        });
      } catch (error) {
        logger.error('Failed to retrieve event', { error: error.message, id: req.params.id });
        res.status(500).json({ error: 'Failed to retrieve event' });
      }
    }
  );

  // Metrics summary endpoint
  router.get('/metrics-summary', (req, res) => {
    try {
      // Get data from our metrics tracker
      const apiCounter = metricsTracker.apiRequestsCounter.value || 0;
      const blockchainCounter = metricsTracker.blockchainTransactionsCounter.value || 0;
      const aiCounter = metricsTracker.aiPredictionsCounter.value || 0;
      
      // Get API latency stats if available
      let apiLatency = { avg: 0 };
      if (metricsTracker.apiLatencyHistogram && metricsTracker.apiLatencyHistogram.getStats) {
        apiLatency = metricsTracker.apiLatencyHistogram.getStats();
      }
      
      res.json({
        api_requests: apiCounter,
        blockchain_transactions: blockchainCounter,
        ai_predictions: aiCounter,
        api_latency: apiLatency.avg || 0,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Failed to generate metrics summary', { error: error.message });
      res.status(500).json({ error: 'Failed to generate metrics summary' });
    }
  });

  return router;
}; 