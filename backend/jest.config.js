/**
 * Jest configuration for NeuraShield backend
 */

module.exports = {
  testEnvironment: 'node',
  collectCoverageFrom: [
    'src/**/*.js',
    '!**/node_modules/**',
  ],
  coverageDirectory: '../coverage/backend',
  testMatch: [
    '**/tests/**/*.test.js',
    '**/test/**/*.test.js',
  ],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  setupFilesAfterEnv: ['../test/setup.js'],
  verbose: true,
}; 