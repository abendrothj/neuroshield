// Set test environment variables
process.env.NODE_ENV = 'test'
process.env.PORT = '3001'
process.env.IPFS_URL = 'http://localhost:5001'
process.env.FABRIC_CA_URL = 'http://localhost:7054'
process.env.FABRIC_CA_NAME = 'ca.org1.example.com'
process.env.FABRIC_CA_ADMIN_USER = 'admin'
process.env.FABRIC_CA_ADMIN_PASSWORD = 'adminpw'

// Mock Fabric Network
jest.mock('fabric-network', () => ({
  Gateway: jest.fn().mockImplementation(() => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    getNetwork: jest.fn().mockResolvedValue({
      getContract: jest.fn().mockResolvedValue({
        submitTransaction: jest.fn(),
        evaluateTransaction: jest.fn(),
      }),
    }),
  })),
  Wallets: {
    newFileSystemWallet: jest.fn(),
  },
}))

// Mock IPFS Client
jest.mock('ipfs-http-client', () => ({
  create: jest.fn().mockReturnValue({
    add: jest.fn(),
    cat: jest.fn(),
  }),
})) 