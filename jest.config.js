module.exports = {
  projects: [
    {
      displayName: 'backend',
      testEnvironment: 'node',
      testMatch: ['<rootDir>/backend/tests/**/*.test.js'],
      transform: {},
      rootDir: '.',
      modulePathIgnorePatterns: ['<rootDir>/node_modules/'],
    },
    {
      displayName: 'integration',
      testEnvironment: 'node',
      testMatch: ['<rootDir>/test/**/*.test.js'],
      transform: {},
      rootDir: '.',
      modulePathIgnorePatterns: ['<rootDir>/node_modules/'],
    },
    {
      displayName: 'frontend',
      testEnvironment: 'jsdom',
      testMatch: ['<rootDir>/frontend/**/*.test.js'],
      transform: {
        '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', { presets: ['next/babel'] }],
      },
      rootDir: '.',
      modulePathIgnorePatterns: ['<rootDir>/node_modules/'],
    }
  ],
  verbose: true,
  collectCoverageFrom: [
    '**/*.{js,jsx}',
    '!**/node_modules/**',
    '!**/vendor/**',
    '!**/coverage/**',
    '!jest.config.js',
    '!**/dist/**',
  ],
  coverageThreshold: {
    global: {
      branches: 60,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
}; 