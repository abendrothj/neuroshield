/**
 * Jest setup file for NeuraShield tests
 */

// Set up environment variables for testing
process.env.NODE_ENV = 'test';
process.env.GCP_PROJECT_ID = 'test-project-id';
process.env.SERVICE_NAME = 'neurashield-test';
process.env.SERVICE_VERSION = '1.0.0-test';

// Mock GCP services
jest.mock('@google-cloud/error-reporting', () => {
  return {
    ErrorReporting: jest.fn().mockImplementation(() => {
      return {
        report: jest.fn(),
      };
    }),
  };
});

jest.mock('@google-cloud/opentelemetry-cloud-trace-exporter', () => {
  return {
    TraceExporter: jest.fn().mockImplementation(() => {
      return {
        export: jest.fn(),
      };
    }),
  };
});

jest.mock('@google-cloud/profiler', () => {
  return {
    start: jest.fn(),
  };
});

// Add global test helpers
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

// Learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock the Next.js router
jest.mock('next/router', () => ({
  useRouter: jest.fn().mockReturnValue({
    push: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
    back: jest.fn(),
    reload: jest.fn(),
    pathname: '/',
    query: {},
    asPath: '/',
    events: {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
    },
    locale: 'en',
    locales: ['en'],
    defaultLocale: 'en',
  }),
}));

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn().mockReturnValue({
    push: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    refresh: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: jest.fn().mockReturnValue('/'),
  useSearchParams: jest.fn().mockReturnValue(new URLSearchParams()),
}));

// Mock for IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe() {
    return null;
  }
  unobserve() {
    return null;
  }
  disconnect() {
    return null;
  }
};

// Reset all mocks after each test
afterEach(() => {
  jest.clearAllMocks();
}); 