import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Dashboard } from '@/components/Dashboard';

export default function Home() {
  return (
    <ProtectedRoute>
      <main className="min-h-screen bg-gray-50">
        <div className="container mx-auto py-8">
          <h1 className="text-4xl font-bold mb-4 text-center">NeuraShield Dashboard</h1>
          <p className="text-lg mb-8 text-center text-gray-600">
            Blockchain-based AI Security Monitoring Platform
          </p>
          <Dashboard />
        </div>
      </main>
    </ProtectedRoute>
  );
} 