# NeuraShield Frontend

This is the frontend application for the NeuraShield blockchain-based AI security monitoring platform.

## Features

- Real-time security event monitoring
- AI detection visualization
- Blockchain transaction tracking
- Mock API for development

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm 9.x or later

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/neurashield.git
   cd neurashield/v0-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   - For development, use `.env.development`
   - For production, use `.env.production`

### Development

Run the development server:

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

By default, the development environment uses mock data. To use the real API:
1. Set `NEXT_PUBLIC_USE_MOCK_API=false` in `.env.development`
2. Update `NEXT_PUBLIC_BLOCKCHAIN_API_URL` to point to your API endpoint

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
v0-frontend/
├── src/
│   ├── app/              # Next.js app router
│   ├── components/       # React components
│   └── lib/
│       ├── api/         # API services
│       ├── hooks/       # Custom React hooks
│       └── types/       # TypeScript types
├── public/              # Static assets
└── .env.*              # Environment configurations
```

## API Integration

The application can work with both mock and real APIs:

- Mock API: Provides sample data for development
- Real API: Connects to the blockchain backend

To switch between them, update the `NEXT_PUBLIC_USE_MOCK_API` environment variable.

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 