import React, { ReactNode } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const router = useRouter();
  
  const isActive = (path: string) => {
    return router.pathname === path ? 'bg-primary-dark text-white' : 'text-gray-700 hover:bg-gray-100';
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/">
                <span className="text-xl font-bold text-primary cursor-pointer">NeuraShield</span>
              </Link>
            </div>
            <nav className="flex">
              <Link href="/dashboard">
                <span className={`inline-flex items-center px-4 cursor-pointer ${isActive('/dashboard')}`}>
                  Dashboard
                </span>
              </Link>
              <Link href="/ai-monitoring">
                <span className={`inline-flex items-center px-4 cursor-pointer ${isActive('/ai-monitoring')}`}>
                  AI Monitoring
                </span>
              </Link>
              <Link href="/events">
                <span className={`inline-flex items-center px-4 cursor-pointer ${isActive('/events')}`}>
                  Threat Events
                </span>
              </Link>
              <Link href="/model-training">
                <span className={`inline-flex items-center px-4 cursor-pointer ${isActive('/model-training')}`}>
                  Model Training
                </span>
              </Link>
            </nav>
          </div>
        </div>
      </header>
      
      <main className="flex-grow bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </div>
      </main>
      
      <footer className="bg-white border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4 text-center text-sm text-gray-500">
            &copy; {new Date().getFullYear()} NeuraShield. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout; 