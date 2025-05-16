/**
 * Local observability implementation for NeuraShield
 * - Logging with Winston
 * - Local metrics tracking
 * - Error handling
 * - Request timing
 */

const winston = require('winston');
const { createServer } = require('http');
const os = require('os');

// Load environment configuration
const SERVICE_NAME = process.env.SERVICE_NAME || 'neurashield-backend';
const SERVICE_VERSION = process.env.SERVICE_VERSION || '1.0.0';
const LOG_LEVEL = process.env.LOG_LEVEL || 'info';

// Initialize logger
const logger = winston.createLogger({
  level: LOG_LEVEL,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { 
    service: SERVICE_NAME,
    version: SERVICE_VERSION,
    hostname: os.hostname()
  },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ 
      filename: 'logs/error.log', 
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    new winston.transports.File({ 
      filename: 'logs/combined.log',
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    })
  ]
});

// Setup metrics tracking
class MetricsTracker {
  constructor() {
    this.counters = new Map();
    this.histograms = new Map();
    this.startTime = Date.now();
    this.lastSnapshotTime = this.startTime;

    // Create standard meters
    this.apiMeter = this.createMeter('api');
    this.blockchainMeter = this.createMeter('blockchain');
    this.aiMeter = this.createMeter('ai');

    // Create standard counters
    this.apiRequestsCounter = this.createCounter('api_requests', {
      description: 'Number of API requests',
    });
    
    this.blockchainTransactionsCounter = this.createCounter('blockchain_transactions', {
      description: 'Number of blockchain transactions',
    });
    
    this.aiPredictionsCounter = this.createCounter('ai_predictions', {
      description: 'Number of AI predictions',
    });
    
    // Create standard histograms
    this.apiLatencyHistogram = this.createHistogram('api_latency', {
      description: 'API request latency',
      unit: 'ms',
    });
    
    this.blockchainLatencyHistogram = this.createHistogram('blockchain_latency', {
      description: 'Blockchain transaction latency',
      unit: 'ms',
    });
    
    this.aiLatencyHistogram = this.createHistogram('ai_latency', {
      description: 'AI prediction latency',
      unit: 'ms',
    });

    // Periodically log metrics snapshot
    this.snapshotInterval = setInterval(() => this.logMetricsSnapshot(), 60000);
  }

  createMeter(name) {
    return { name, counters: [], histograms: [] };
  }

  createCounter(name, options = {}) {
    if (this.counters.has(name)) {
      return this.counters.get(name);
    }

    const counter = {
      name,
      description: options.description || '',
      value: 0,
      labels: [],
      add: (value, labels = {}) => {
        counter.value += value;
        counter.labels.push({ ...labels, value, timestamp: new Date() });
        logger.debug(`Counter ${name} incremented by ${value}`, { counter: name, value, labels });
        return counter.value;
      }
    };

    this.counters.set(name, counter);
    return counter;
  }

  createHistogram(name, options = {}) {
    if (this.histograms.has(name)) {
      return this.histograms.get(name);
    }

    const histogram = {
      name,
      description: options.description || '',
      unit: options.unit || '',
      records: [],
      record: (value, labels = {}) => {
        const record = { value, labels, timestamp: new Date() };
        histogram.records.push(record);
        
        // Keep only the last 1000 records to prevent memory issues
        if (histogram.records.length > 1000) {
          histogram.records.shift();
        }
        
        logger.debug(`Histogram ${name} recorded value ${value}`, { histogram: name, value, labels });
        return histogram.records.length;
      },
      getStats: () => {
        if (histogram.records.length === 0) return { count: 0 };
        
        const values = histogram.records.map(r => r.value);
        const sum = values.reduce((a, b) => a + b, 0);
        const count = values.length;
        const avg = sum / count;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        // Calculate percentiles
        const sorted = [...values].sort((a, b) => a - b);
        const p50 = sorted[Math.floor(count * 0.5)];
        const p95 = sorted[Math.floor(count * 0.95)];
        const p99 = sorted[Math.floor(count * 0.99)];
        
        return { count, sum, avg, min, max, p50, p95, p99 };
      }
    };

    this.histograms.set(name, histogram);
    return histogram;
  }

  logMetricsSnapshot() {
    const now = Date.now();
    const uptime = (now - this.startTime) / 1000;
    const timeSinceLastSnapshot = (now - this.lastSnapshotTime) / 1000;
    this.lastSnapshotTime = now;

    const countersSnapshot = {};
    this.counters.forEach((counter, name) => {
      const recentLabels = counter.labels
        .filter(l => (now - l.timestamp) <= 60000)
        .length;
        
      countersSnapshot[name] = {
        total: counter.value,
        recentCount: recentLabels,
        ratePerMinute: recentLabels / (timeSinceLastSnapshot / 60)
      };
    });

    const histogramsSnapshot = {};
    this.histograms.forEach((histogram, name) => {
      const recentRecords = histogram.records
        .filter(r => (now - r.timestamp) <= 60000);
        
      histogramsSnapshot[name] = {
        records: histogram.records.length,
        recentRecords: recentRecords.length,
        stats: histogram.getStats()
      };
    });

    logger.info('Metrics snapshot', {
      uptime,
      counters: countersSnapshot,
      histograms: histogramsSnapshot
    });
  }
}

// Setup error handling
class ErrorHandler {
  constructor() {
    this.errors = [];
  }

  report(error, request = null) {
    const errorInfo = {
      message: error.message,
      stack: error.stack,
      timestamp: new Date(),
      reqInfo: request ? {
        method: request.method,
        url: request.url,
        ip: request.ip,
        userAgent: request.get('user-agent')
      } : null
    };
    
    this.errors.push(errorInfo);
    
    // Keep only the last 100 errors to prevent memory issues
    if (this.errors.length > 100) {
      this.errors.shift();
    }
    
    logger.error('Error occurred', {
      error: {
        message: error.message,
        stack: error.stack,
        name: error.name
      },
      request: errorInfo.reqInfo
    });
    
    return errorInfo;
  }

  getRecentErrors(count = 10) {
    return this.errors.slice(-count);
  }
}

// Create metrics server for exposing metrics endpoint
function createMetricsServer(port = 9100, metricsTracker) {
  const server = createServer((req, res) => {
    if (req.url === '/metrics') {
      const countersData = {};
      metricsTracker.counters.forEach((counter, name) => {
        countersData[name] = {
          value: counter.value,
          description: counter.description,
        };
      });

      const histogramsData = {};
      metricsTracker.histograms.forEach((histogram, name) => {
        histogramsData[name] = {
          description: histogram.description,
          unit: histogram.unit,
          stats: histogram.getStats()
        };
      });

      const metrics = {
        service: SERVICE_NAME,
        version: SERVICE_VERSION,
        timestamp: new Date(),
        counters: countersData,
        histograms: histogramsData
      };

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(metrics, null, 2));
      return;
    }

    res.writeHead(404);
    res.end('Not found');
  });

  server.listen(port, () => {
    logger.info(`Metrics server listening on port ${port}`);
  });

  return server;
}

// Express middleware for error reporting
function errorReportingMiddleware(errorHandler) {
  if (!errorHandler) {
    return (err, req, res, next) => next(err);
  }

  return (err, req, res, next) => {
    errorHandler.report(err, req);
    next(err);
  };
}

// Express middleware for API metrics
function metricsMiddleware(metricsTracker) {
  if (!metricsTracker) {
    return (req, res, next) => next();
  }
  
  return (req, res, next) => {
    const startTime = Date.now();
    
    // Add request logger
    req.log = (level, message, meta = {}) => {
      logger[level](message, {
        ...meta,
        req: {
          method: req.method,
          url: req.url,
          path: req.path
        }
      });
    };
    
    // Record API request
    metricsTracker.apiRequestsCounter.add(1, {
      method: req.method,
      path: req.route ? req.route.path : req.path,
      status: res.statusCode,
    });
    
    // Record latency on response
    const originalEnd = res.end;
    res.end = function(...args) {
      const endTime = Date.now();
      const latency = endTime - startTime;
      
      metricsTracker.apiLatencyHistogram.record(latency, {
        method: req.method,
        path: req.route ? req.route.path : req.path,
        status: res.statusCode,
      });
      
      // Log request completion
      logger.http(`${req.method} ${req.url}`, {
        method: req.method,
        url: req.url,
        status: res.statusCode,
        latency,
        ip: req.ip || req.connection.remoteAddress
      });
      
      originalEnd.apply(res, args);
    };
    
    next();
  };
}

// Initialize all observability services
function initObservability(options = {}) {
  const metricsTracker = new MetricsTracker();
  const errorHandler = new ErrorHandler();
  
  // Create metrics server if enabled
  let metricsServer = null;
  if (options.enableMetricsServer !== false) {
    const metricsPort = options.metricsPort || 9100;
    metricsServer = createMetricsServer(metricsPort, metricsTracker);
  }
  
  return {
    logger,
    errorHandler,
    metricsTracker,
    metricsServer,
    middleware: {
      errorReporting: errorReportingMiddleware(errorHandler),
      metrics: metricsMiddleware(metricsTracker)
    }
  };
}

module.exports = {
  initObservability,
  logger
}; 