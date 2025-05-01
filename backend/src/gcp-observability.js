/**
 * Google Cloud Platform observability integration for NeuraShield
 * - Cloud Trace for distributed tracing
 * - Cloud Profiler for performance analysis
 * - Error Reporting for error tracking
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

// Initialize all observability services
function initObservability() {
  const { trace } = setupCloudTrace();
  setupCloudProfiler();
  const { errorReporting } = setupErrorReporting();
  
  return {
    trace,
    errorReporting,
    errorReportingMiddleware: errorReportingMiddleware(errorReporting)
  };
}

module.exports = {
  initObservability
}; 