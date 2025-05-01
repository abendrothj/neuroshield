import React, { useState } from 'react';
import { BlockchainEvents } from './BlockchainEvents';
import { AIDetections } from './AIDetections';

type Tab = 'events' | 'ai-detections' | 'metrics';

export function Dashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('events');

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('events')}
              className={`py-4 px-1 ${
                activeTab === 'events'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Security Events
            </button>
            <button
              onClick={() => setActiveTab('ai-detections')}
              className={`py-4 px-1 ${
                activeTab === 'ai-detections'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              AI Detections
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`py-4 px-1 ${
                activeTab === 'metrics'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Metrics & Analytics
            </button>
          </nav>
        </div>
      </div>

      <div className="mt-6">
        {activeTab === 'events' && <BlockchainEvents />}
        {activeTab === 'ai-detections' && <AIDetections />}
        {activeTab === 'metrics' && (
          <div className="p-4 bg-gray-50 rounded-lg text-center">
            <h2 className="text-2xl font-bold mb-4">Metrics & Analytics</h2>
            <p className="text-gray-500">
              Analytics dashboard coming soon. This will display blockchain metrics, event statistics,
              and AI model performance data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
} 