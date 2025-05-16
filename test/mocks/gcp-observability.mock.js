/**
 * Mock GCP Observability for testing
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

class MockErrorReporting {
  constructor() {
    this.errors = [];
  }

  report(error) {
    this.errors.push({
      error,
      timestamp: new Date(),
    });
  }
}

function initObservability() {
  // Create mock meter provider
  const apiMeter = new MockMeter('api');
  const blockchainMeter = new MockMeter('blockchain');
  const aiMeter = new MockMeter('ai');
  
  // Create counters for key metrics
  const apiRequestsCounter = apiMeter.createCounter('api_requests', {
    description: 'Number of API requests',
  });
  
  const blockchainTransactionsCounter = blockchainMeter.createCounter('blockchain_transactions', {
    description: 'Number of blockchain transactions',
  });
  
  const aiPredictionsCounter = aiMeter.createCounter('ai_predictions', {
    description: 'Number of AI predictions',
  });
  
  // Create histograms for latency metrics
  const apiLatencyHistogram = apiMeter.createHistogram('api_latency', {
    description: 'API request latency',
    unit: 'ms',
  });
  
  const blockchainLatencyHistogram = blockchainMeter.createHistogram('blockchain_latency', {
    description: 'Blockchain transaction latency',
    unit: 'ms',
  });
  
  const aiLatencyHistogram = aiMeter.createHistogram('ai_latency', {
    description: 'AI prediction latency',
    unit: 'ms',
  });

  // Create error reporting
  const errorReporting = new MockErrorReporting();

  // Express middleware for Error Reporting
  const errorReportingMiddleware = (err, req, res, next) => {
    errorReporting.report(err);
    next(err);
  };

  // Express middleware for API metrics
  const metricsMiddleware = (req, res, next) => {
    const startTime = Date.now();
    
    // Record API request
    apiRequestsCounter.add(1, {
      method: req.method,
      path: req.route ? req.route.path : req.path,
      status: res.statusCode,
    });
    
    // Record latency on response
    const originalEnd = res.end;
    res.end = function(...args) {
      const endTime = Date.now();
      const latency = endTime - startTime;
      
      apiLatencyHistogram.record(latency, {
        method: req.method,
        path: req.route ? req.route.path : req.path,
        status: res.statusCode,
      });
      
      originalEnd.apply(res, args);
    };
    
    next();
  };
  
  return {
    trace: { mock: true },
    errorReporting,
    monitoring: {
      mock: true,
      meterProvider: { mock: true },
      meters: {
        apiMeter,
        blockchainMeter,
        aiMeter
      },
      counters: {
        apiRequestsCounter,
        blockchainTransactionsCounter,
        aiPredictionsCounter
      },
      histograms: {
        apiLatencyHistogram,
        blockchainLatencyHistogram,
        aiLatencyHistogram
      }
    },
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
  MockErrorReporting
}; 