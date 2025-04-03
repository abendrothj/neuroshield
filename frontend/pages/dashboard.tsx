import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/ui/loading-spinner';
import Alert from '../components/ui/alert';

interface ThreatData {
  id: string;
  timestamp: string;
  type: string;
  details: {
    threat_level: string;
    source: string;
    target: string;
  };
  ipfsHash: string;
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [threatData, setThreatData] = useState<ThreatData[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/events');
        
        if (!response.ok) {
          throw new Error(`Error fetching data: ${response.statusText}`);
        }
        
        const data = await response.json();
        setThreatData(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching threat data:', err);
        setError('Failed to load threat data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Set up polling every 30 seconds
    const interval = setInterval(fetchData, 30000);
    
    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Threat Detection Dashboard</h1>
      
      {error && (
        <Alert type="error" message={error} className="mb-4" />
      )}
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <div>
          {threatData.length === 0 ? (
            <Alert type="info" message="No threat data available." className="mb-4" />
          ) : (
            <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
              {threatData.map((threat) => (
                <div 
                  key={threat.id}
                  className="border rounded-lg p-4 shadow-sm"
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold">{threat.type}</h3>
                    <span className={`px-2 py-1 rounded text-xs ${
                      threat.details.threat_level === 'critical' ? 'bg-red-100 text-red-800' :
                      threat.details.threat_level === 'high_risk' ? 'bg-orange-100 text-orange-800' :
                      threat.details.threat_level === 'medium_risk' ? 'bg-yellow-100 text-yellow-800' :
                      threat.details.threat_level === 'low_risk' ? 'bg-blue-100 text-blue-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {threat.details.threat_level.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <div className="text-sm text-gray-500 mb-2">
                    {new Date(threat.timestamp).toLocaleString()}
                  </div>
                  
                  <div className="text-sm mb-2">
                    <div><span className="font-medium">Source:</span> {threat.details.source}</div>
                    <div><span className="font-medium">Target:</span> {threat.details.target}</div>
                  </div>
                  
                  <div className="text-xs text-gray-400 truncate">
                    IPFS: {threat.ipfsHash}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Dashboard; 