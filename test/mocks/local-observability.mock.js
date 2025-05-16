/**
 * Mock Local Observability for testing
 */

class MockCounter {
  constructor(name, options = {}) {
    this.name = name;
    this.description = options.description || '';
    this.value = 0;
    this.labels = [];
  }

  add(value, labels = {}) {
    this.value += value;
    this.labels.push({ ...labels, value, timestamp: new Date() });
    return this.value;
  }
}

class MockHistogram {
  constructor(name, options = {}) {
    this.name = name;
    this.description = options.description || '';
    this.unit = options.unit || '';
    this.records = [];
  }

  record(value, labels = {}) {
    this.records.push({ value, labels, timestamp: new Date() });
    return this.records.length;
  }
  
  getStats() {
    if (this.records.length === 0) return { count: 0 };
    
    const values = this.records.map(r => r.value);
    const sum = values.reduce((a, b) => a + b, 0);
    const count = values.length;
    const avg = sum / count;
    
    return { count, sum, avg };
  }
}

class MockMeter {
  constructor(name) {
    this.name = name;
    this.counters = new Map();
    this.histograms = new Map();
  }

  createCounter(name, options = {}) {
    const counter = new MockCounter(name, options);
    this.counters.set(name, counter);
    return counter;
  }

  createHistogram(name, options = {}) {
    const histogram = new MockHistogram(name, options);
    this.histograms.set(name, histogram);
    return histogram;
  }
}

class MockErrorHandler {
  constructor() {
    this.errors = [];
  }

  report(error, request = null) {
    this.errors.push({
      error,
      request,
      timestamp: new Date(),
    });
    return this.errors.length;
  }
  
  getRecentErrors(count = 10) {
    return this.errors.slice(-count);
  }
}

class MockMetricsTracker {
  constructor() {
    this.counters = new Map();
    this.histograms = new Map();
    
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
  }
  
  createMeter(name) {
    return { name, counters: [], histograms: [] };
  }

  createCounter(name, options = {}) {
    if (this.counters.has(name)) {
      return this.counters.get(name);
    }
    
    const counter = new MockCounter(name, options);
    this.counters.set(name, counter);
    return counter;
  }

  createHistogram(name, options = {}) {
    if (this.histograms.has(name)) {
      return this.histograms.get(name);
    }
    
    const histogram = new MockHistogram(name, options);
    this.histograms.set(name, histogram);
    return histogram;
  }
}

const mockLogger = {
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn(),
  http: jest.fn(),
  debug: jest.fn(),
  silly: jest.fn()
};

function initObservability() {
  const metricsTracker = new MockMetricsTracker();
  const errorHandler = new MockErrorHandler();
  
  // Express middleware for Error Reporting
  const errorReportingMiddleware = (err, req, res, next) => {
    errorHandler.report(err, req);
    next(err);
  };

  // Express middleware for API metrics
  const metricsMiddleware = (req, res, next) => {
    const startTime = Date.now();
    
    // Add request logger
    req.log = (level, message, meta = {}) => {
      mockLogger[level](message, meta);
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
      
      originalEnd.apply(res, args);
    };
    
    next();
  };
  
  return {
    logger: mockLogger,
    errorHandler,
    metricsTracker,
    middleware: {
      errorReporting: errorReportingMiddleware,
      metrics: metricsMiddleware
    }
  };
}

module.exports = {
  initObservability,
  MockMeter,
  MockCounter,
  MockHistogram,
  MockErrorHandler,
  MockMetricsTracker,
  mockLogger
}; 