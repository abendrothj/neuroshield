/**
 * Google Cloud Platform observability integration for NeuraShield
 * - Cloud Trace for distributed tracing
 * - Cloud Profiler for performance analysis
 * - Error Reporting for error tracking
 * - Cloud Monitoring for metrics
 */

// Load environment configuration
const PROJECT_ID = process.env.GCP_PROJECT_ID;
const SERVICE_NAME = process.env.SERVICE_NAME || 'neurashield-backend';
const SERVICE_VERSION = process.env.SERVICE_VERSION || '1.0.0';

// Configure Cloud Trace
function setupCloudTrace() {
  if (!PROJECT_ID) {
    console.warn('GCP_PROJECT_ID not set. Cloud Trace will not be activated.');
    return { trace: null };
  }

  try {
    // Initialize Cloud Trace
    const { TraceExporter } = require('@google-cloud/opentelemetry-cloud-trace-exporter');
    const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
    const { registerInstrumentations } = require('@opentelemetry/instrumentation');
    const { SimpleSpanProcessor } = require('@opentelemetry/sdk-trace-base');
    const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
    const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
    const { Resource } = require('@opentelemetry/resources');
    const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');

    // Create a tracer provider
    const provider = new NodeTracerProvider({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: SERVICE_NAME,
        [SemanticResourceAttributes.SERVICE_VERSION]: SERVICE_VERSION,
      }),
    });

    // Configure exporter to send traces to Cloud Trace
    const exporter = new TraceExporter({
      projectId: PROJECT_ID,
    });

    // Register the exporter
    provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    provider.register();

    // Register instrumentations for Express and HTTP
    registerInstrumentations({
      instrumentations: [
        new HttpInstrumentation(),
        new ExpressInstrumentation(),
      ],
    });

    console.log('Cloud Trace initialized successfully');
    return { trace: provider };
  } catch (error) {
    console.error('Failed to initialize Cloud Trace:', error);
    return { trace: null };
  }
}

// Configure Cloud Profiler
function setupCloudProfiler() {
  if (!PROJECT_ID) {
    console.warn('GCP_PROJECT_ID not set. Cloud Profiler will not be activated.');
    return;
  }

  try {
    // Initialize Cloud Profiler
    const profiler = require('@google-cloud/profiler');
    profiler.start({
      projectId: PROJECT_ID,
      serviceContext: {
        service: SERVICE_NAME,
        version: SERVICE_VERSION,
      },
    });
    console.log('Cloud Profiler initialized successfully');
  } catch (error) {
    console.error('Failed to initialize Cloud Profiler:', error);
  }
}

// Configure Cloud Monitoring
function setupCloudMonitoring() {
  if (!PROJECT_ID) {
    console.warn('GCP_PROJECT_ID not set. Cloud Monitoring will not be activated.');
    return { monitoring: null };
  }

  try {
    // Initialize Cloud Monitoring
    const { MetricExporter } = require('@google-cloud/opentelemetry-cloud-monitoring-exporter');
    const { MeterProvider } = require('@opentelemetry/sdk-metrics');
    const { Resource } = require('@opentelemetry/resources');
    const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
    
    // Create a meter provider
    const meterProvider = new MeterProvider({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: SERVICE_NAME,
        [SemanticResourceAttributes.SERVICE_VERSION]: SERVICE_VERSION,
      }),
    });
    
    // Configure exporter to send metrics to Cloud Monitoring
    const exporter = new MetricExporter({
      projectId: PROJECT_ID,
    });
    
    // Register the exporter
    meterProvider.addMetricExporter(exporter);
    
    // Create meters for specific usage
    const apiMeter = meterProvider.getMeter('api');
    const blockchainMeter = meterProvider.getMeter('blockchain');
    const aiMeter = meterProvider.getMeter('ai');
    
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
    
    console.log('Cloud Monitoring initialized successfully');
    return { 
      monitoring: {
        meterProvider,
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
      }
    };
  } catch (error) {
    console.error('Failed to initialize Cloud Monitoring:', error);
    return { monitoring: null };
  }
}

// Configure Error Reporting
function setupErrorReporting() {
  if (!PROJECT_ID) {
    console.warn('GCP_PROJECT_ID not set. Error Reporting will not be activated.');
    return { errorReporting: null };
  }

  try {
    // Initialize Error Reporting
    const { ErrorReporting } = require('@google-cloud/error-reporting');
    const errorReporting = new ErrorReporting({
      projectId: PROJECT_ID,
      reportMode: 'production',
      serviceContext: {
        service: SERVICE_NAME,
        version: SERVICE_VERSION,
      },
    });
    
    console.log('Error Reporting initialized successfully');
    return { errorReporting };
  } catch (error) {
    console.error('Failed to initialize Error Reporting:', error);
    return { errorReporting: null };
  }
}

// Express middleware for Error Reporting
function errorReportingMiddleware(errorReporting) {
  if (!errorReporting) {
    return (err, req, res, next) => next(err);
  }

  return (err, req, res, next) => {
    errorReporting.report(err);
    next(err);
  };
}

// Express middleware for API metrics
function metricsMiddleware(monitoring) {
  if (!monitoring || !monitoring.counters || !monitoring.histograms) {
    return (req, res, next) => next();
  }
  
  return (req, res, next) => {
    const startTime = Date.now();
    
    // Record API request
    monitoring.counters.apiRequestsCounter.add(1, {
      method: req.method,
      path: req.route ? req.route.path : req.path,
      status: res.statusCode,
    });
    
    // Record latency on response
    const originalEnd = res.end;
    res.end = function(...args) {
      const endTime = Date.now();
      const latency = endTime - startTime;
      
      monitoring.histograms.apiLatencyHistogram.record(latency, {
        method: req.method,
        path: req.route ? req.route.path : req.path,
        status: res.statusCode,
      });
      
      originalEnd.apply(res, args);
    };
    
    next();
  };
}

// Initialize all observability services
function initObservability() {
  const { trace } = setupCloudTrace();
  setupCloudProfiler();
  const { errorReporting } = setupErrorReporting();
  const { monitoring } = setupCloudMonitoring();
  
  return {
    trace,
    errorReporting,
    monitoring,
    middleware: {
      errorReporting: errorReportingMiddleware(errorReporting),
      metrics: metricsMiddleware(monitoring)
    }
  };
}

module.exports = {
  initObservability
}; 