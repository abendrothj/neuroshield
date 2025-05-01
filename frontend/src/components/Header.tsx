import React from 'react';
import { useAuth } from '@/lib/contexts/AuthContext';

export function Header() {
  const { user, logout } = useAuth();

  return (
    <header className="bg-white shadow">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <div className="flex items-center">
          <h1 className="text-xl font-bold text-gray-900">NeuraShield</h1>
        </div>
        
        {user && (
          <div className="flex items-center">
            <div className="mr-4 text-sm text-gray-700">
              <span className="font-medium">Logged in as:</span> {user.username}
            </div>
            <button
              onClick={logout}
              className="px-3 py-1 text-sm text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
} 