import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/ui/loading-spinner';
import Alert from '../components/ui/alert';

interface ModelMetrics {
  accuracy: number;
  inference_time: number;
  memory_usage: number;
  predictions_total: number;
  error_rate: number;
}

const AiMonitoring: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        // In a real application, this would fetch from a metrics API
        // This is a mock implementation
        const response = await fetch('/api/ai-metrics');
        
        if (!response.ok) {
          throw new Error(`Error fetching metrics: ${response.statusText}`);
        }
        
        const data = await response.json();
        setMetrics(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching AI metrics:', err);
        
        // For demo, use mock data if fetch fails
        setMetrics({
          accuracy: 0.95,
          inference_time: 0.042,
          memory_usage: 453012480, // ~432 MB
          predictions_total: 12453,
          error_rate: 0.02
        });
        
        setError('Using demo data: Could not connect to metrics service');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    
    // Poll for updates
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatMemory = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">AI Model Monitoring</h1>
      
      {error && (
        <Alert type="warning" message={error} className="mb-4" />
      )}
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner size="lg" />
        </div>
      ) : metrics ? (
        <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {/* Accuracy Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-2">Model Accuracy</h2>
            <div className="flex items-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center bg-blue-100 mr-4">
                <span className="text-blue-800 text-xl font-bold">{(metrics.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div>
                <p className="text-gray-500">Current model accuracy</p>
                <div className={`text-sm font-medium ${
                  metrics.accuracy >= 0.9 ? 'text-green-600' : 
                  metrics.accuracy >= 0.8 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {metrics.accuracy >= 0.9 ? 'Excellent' : 
                   metrics.accuracy >= 0.8 ? 'Good' : 'Needs improvement'}
                </div>
              </div>
            </div>
          </div>
          
          {/* Inference Time Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-2">Inference Time</h2>
            <div className="flex items-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center bg-green-100 mr-4">
                <span className="text-green-800 text-xl font-bold">{(metrics.inference_time * 1000).toFixed(0)}ms</span>
              </div>
              <div>
                <p className="text-gray-500">Average prediction time</p>
                <div className={`text-sm font-medium ${
                  metrics.inference_time < 0.05 ? 'text-green-600' : 
                  metrics.inference_time < 0.1 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {metrics.inference_time < 0.05 ? 'Fast' : 
                   metrics.inference_time < 0.1 ? 'Acceptable' : 'Slow'}
                </div>
              </div>
            </div>
          </div>
          
          {/* Memory Usage Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-2">Memory Usage</h2>
            <div className="flex items-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center bg-purple-100 mr-4">
                <span className="text-purple-800 text-xl font-bold">{formatMemory(metrics.memory_usage)}</span>
              </div>
              <div>
                <p className="text-gray-500">Current memory consumption</p>
                <div className="text-sm font-medium text-purple-600">
                  {metrics.memory_usage < 500 * 1024 * 1024 ? 'Optimized' : 'Normal usage'}
                </div>
              </div>
            </div>
          </div>
          
          {/* Predictions Total Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-2">Total Predictions</h2>
            <div className="flex items-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center bg-gray-100 mr-4">
                <span className="text-gray-800 text-xl font-bold">{metrics.predictions_total.toLocaleString()}</span>
              </div>
              <div>
                <p className="text-gray-500">Predictions processed</p>
                <div className="text-sm font-medium text-gray-600">
                  Lifetime count
                </div>
              </div>
            </div>
          </div>
          
          {/* Error Rate Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-2">Error Rate</h2>
            <div className="flex items-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center bg-red-100 mr-4">
                <span className="text-red-800 text-xl font-bold">{(metrics.error_rate * 100).toFixed(1)}%</span>
              </div>
              <div>
                <p className="text-gray-500">Processing errors</p>
                <div className={`text-sm font-medium ${
                  metrics.error_rate < 0.01 ? 'text-green-600' : 
                  metrics.error_rate < 0.05 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {metrics.error_rate < 0.01 ? 'Minimal' : 
                   metrics.error_rate < 0.05 ? 'Acceptable' : 'High'}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <Alert type="error" message="Failed to load metrics data" className="mb-4" />
      )}
    </div>
  );
};

export default AiMonitoring; 