import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

const Home: React.FC = () => {
  const router = useRouter();

  return (
    <div className="flex flex-col space-y-12">
      <section className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">NeuraShield</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Advanced cybersecurity platform combining AI threat detection with blockchain immutability for comprehensive protection.
        </p>
      </section>
      
      <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="card">
          <div className="text-primary mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold mb-2">AI Threat Detection</h2>
          <p className="text-gray-600 mb-4">
            Advanced machine learning models detect and classify cybersecurity threats in real-time.
          </p>
          <Link href="/ai-monitoring">
            <button className="btn btn-primary">Monitor AI</button>
          </Link>
        </div>
        
        <div className="card">
          <div className="text-primary mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold mb-2">Real-time Dashboard</h2>
          <p className="text-gray-600 mb-4">
            Visualize and monitor your network security status with live updates and insights.
          </p>
          <Link href="/dashboard">
            <button className="btn btn-primary">View Dashboard</button>
          </Link>
        </div>
        
        <div className="card">
          <div className="text-primary mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold mb-2">Blockchain Immutability</h2>
          <p className="text-gray-600 mb-4">
            All security events are recorded on blockchain for tamper-proof audit trails.
          </p>
          <Link href="/events">
            <button className="btn btn-primary">Security Events</button>
          </Link>
        </div>
      </section>
      
      <section className="bg-white p-8 rounded-lg shadow-sm">
        <h2 className="text-2xl font-bold mb-4">Getting Started</h2>
        <div className="space-y-4">
          <p className="text-gray-600">
            To get started with NeuraShield, navigate to the dashboard to view your security status or check the AI monitoring page to see the performance of our threat detection system.
          </p>
          <div className="flex space-x-4">
            <button className="btn btn-primary" onClick={() => router.push('/dashboard')}>
              Go to Dashboard
            </button>
            <button className="btn btn-secondary" onClick={() => router.push('/ai-monitoring')}>
              Check AI Status
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home; 