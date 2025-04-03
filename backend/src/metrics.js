const client = require('prom-client');
const express = require('express');

// Create a Registry to register metrics
const register = new client.Registry();

// Add default metrics
client.collectDefaultMetrics({
    register,
    prefix: 'neurashield_',
    timeout: 10000,
    gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5]
});

// Create custom metrics
const httpRequestDuration = new client.Histogram({
    name: 'http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
});

const httpRequestsTotal = new client.Counter({
    name: 'http_requests_total',
    help: 'Total number of HTTP requests',
    labelNames: ['method', 'route', 'status_code']
});

const blockchainSyncStatus = new client.Gauge({
    name: 'blockchain_sync_status',
    help: 'Blockchain sync status (1 = synced, 0 = not synced)'
});

const modelAccuracy = new client.Gauge({
    name: 'model_accuracy',
    help: 'Current accuracy of the threat detection model'
});

// Register custom metrics
register.registerMetric(httpRequestDuration);
register.registerMetric(httpRequestsTotal);
register.registerMetric(blockchainSyncStatus);
register.registerMetric(modelAccuracy);

// Create metrics endpoint
const metricsApp = express();

metricsApp.get('/metrics', async (req, res) => {
    try {
        res.set('Content-Type', register.contentType);
        res.end(await register.metrics());
    } catch (error) {
        res.status(500).end(error);
    }
});

module.exports = {
    register,
    httpRequestDuration,
    httpRequestsTotal,
    blockchainSyncStatus,
    modelAccuracy,
    metricsApp
}; 