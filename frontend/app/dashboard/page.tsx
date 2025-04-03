'use client';

import { useEffect, useState } from 'react';
import Dashboard from "@/components/dashboard"
import Layout from "@/components/layout"

export default function DashboardPage() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <Layout>
      <div className={`transition-opacity duration-500 ${isLoaded ? "opacity-100" : "opacity-0"}`}>
        <Dashboard />
      </div>
    </Layout>
  );
} 