import React, { useState, useMemo } from 'react';
import { useBlockchain } from '@/lib/hooks/useBlockchain';
import { AIDetection } from '@/lib/types/blockchain';

// Define confidence level thresholds and colors
const confidenceColors = {
  high: 'bg-red-500',
  medium: 'bg-yellow-500',
  low: 'bg-blue-500',
};

const getConfidenceLevel = (confidence: number) => {
  if (confidence >= 0.8) return { level: 'high', color: confidenceColors.high };
  if (confidence >= 0.5) return { level: 'medium', color: confidenceColors.medium };
  return { level: 'low', color: confidenceColors.low };
};

const statusColors = {
  PENDING: 'bg-gray-100 text-gray-800',
  CONFIRMED: 'bg-green-100 text-green-800',
  FAILED: 'bg-red-100 text-red-800',
};

type SortField = 'timestamp' | 'confidence' | 'prediction';
type SortDirection = 'asc' | 'desc';
type ConfidenceFilter = 'ALL' | 'HIGH' | 'MEDIUM' | 'LOW';

export function AIDetections() {
  const { aiDetections, fetchAIDetections } = useBlockchain();
  const [searchTerm, setSearchTerm] = useState('');
  const [confidenceFilter, setConfidenceFilter] = useState<ConfidenceFilter>('ALL');
  const [modelFilter, setModelFilter] = useState<string>('ALL');
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const uniqueModels = useMemo(() => {
    if (!aiDetections.data) return [];
    const models = new Set(aiDetections.data.map(detection => detection.modelId));
    return Array.from(models);
  }, [aiDetections.data]);

  const filteredAndSortedDetections = useMemo(() => {
    if (!aiDetections.data) return [];

    // Filter by search term, confidence, and model
    let filtered = aiDetections.data.filter((detection) => {
      const matchesSearch = searchTerm === '' || 
        detection.prediction.toLowerCase().includes(searchTerm.toLowerCase()) ||
        detection.modelId.toLowerCase().includes(searchTerm.toLowerCase());
      
      let matchesConfidence = true;
      if (confidenceFilter !== 'ALL') {
        const { level } = getConfidenceLevel(detection.confidence);
        matchesConfidence = level.toUpperCase() === confidenceFilter;
      }
      
      const matchesModel = modelFilter === 'ALL' || detection.modelId === modelFilter;
      
      return matchesSearch && matchesConfidence && matchesModel;
    });

    // Sort the filtered detections
    return filtered.sort((a, b) => {
      let comparison = 0;
      
      if (sortField === 'timestamp') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortField === 'confidence') {
        comparison = a.confidence - b.confidence;
      } else if (sortField === 'prediction') {
        comparison = a.prediction.localeCompare(b.prediction);
      }
      
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [aiDetections.data, searchTerm, confidenceFilter, modelFilter, sortField, sortDirection]);

  if (aiDetections.loading) {
    return <div className="p-4">Loading AI detections...</div>;
  }

  if (aiDetections.error) {
    return (
      <div className="p-4 text-red-600">
        Error: {aiDetections.error}
        <button
          onClick={fetchAIDetections}
          className="ml-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!aiDetections.data || aiDetections.data.length === 0) {
    return <div className="p-4">No AI detections found.</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <h2 className="text-2xl font-bold">AI Detections</h2>
        <div className="flex flex-col md:flex-row gap-2 w-full md:w-auto">
          <input
            type="text"
            placeholder="Search detections..."
            className="px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          
          <select
            className="px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={confidenceFilter}
            onChange={(e) => setConfidenceFilter(e.target.value as ConfidenceFilter)}
          >
            <option value="ALL">All Confidence Levels</option>
            <option value="HIGH">High (&gt; 80%)</option>
            <option value="MEDIUM">Medium (50% - 80%)</option>
            <option value="LOW">Low (&lt; 50%)</option>
          </select>
          
          <select
            className="px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
          >
            <option value="ALL">All Models</option>
            {uniqueModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="overflow-x-auto bg-white rounded-lg shadow">
        <div className="flex items-center justify-between p-4 border-b">
          <div className="text-sm text-gray-500">
            Showing {filteredAndSortedDetections.length} of {aiDetections.data.length} detections
          </div>
          <div className="flex gap-2">
            <button 
              onClick={() => handleSort('timestamp')}
              className={`px-2 py-1 text-xs rounded ${sortField === 'timestamp' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}
            >
              Time {sortField === 'timestamp' && (sortDirection === 'asc' ? '↑' : '↓')}
            </button>
            <button 
              onClick={() => handleSort('confidence')}
              className={`px-2 py-1 text-xs rounded ${sortField === 'confidence' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}
            >
              Confidence {sortField === 'confidence' && (sortDirection === 'asc' ? '↑' : '↓')}
            </button>
            <button 
              onClick={() => handleSort('prediction')}
              className={`px-2 py-1 text-xs rounded ${sortField === 'prediction' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}
            >
              Prediction {sortField === 'prediction' && (sortDirection === 'asc' ? '↑' : '↓')}
            </button>
          </div>
        </div>
        
        <div className="grid gap-4 p-4">
          {filteredAndSortedDetections.map((detection: AIDetection) => {
            const { level, color } = getConfidenceLevel(detection.confidence);
            return (
              <div
                key={detection.id}
                className="border rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold">
                    {detection.prediction}
                  </h3>
                  <div className="flex gap-2">
                    <span className={`px-2 py-1 rounded text-sm ${statusColors[detection.status]}`}>
                      {detection.status}
                    </span>
                  </div>
                </div>
                
                <div className="mt-3 mb-3">
                  <div className="flex items-center">
                    <span className="text-sm mr-2">Confidence: {Math.round(detection.confidence * 100)}%</span>
                    <div className="h-2 flex-grow rounded-full bg-gray-200">
                      <div 
                        className={`h-2 rounded-full ${color}`} 
                        style={{ width: `${detection.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="ml-2 text-xs uppercase px-2 rounded bg-gray-100">{level}</span>
                  </div>
                </div>
                
                <div className="text-sm text-gray-600">
                  <p>Model: {detection.modelId}</p>
                  <p>Type: {detection.metadata.detectionType}</p>
                  <p>Processing Time: {detection.metadata.processingTime}ms</p>
                  <p>Time: {new Date(detection.timestamp).toLocaleString()}</p>
                  {detection.blockchainTxHash && (
                    <p className="font-mono text-xs mt-2">
                      TX: {detection.blockchainTxHash}
                    </p>
                  )}
                </div>
                
                <div className="mt-2">
                  <div className="flex flex-col">
                    <p className="text-sm font-semibold mb-1">Input Data:</p>
                    <div className="bg-gray-50 p-2 rounded text-xs">
                      <pre className="overflow-auto max-h-32">
                        {JSON.stringify(detection.inputData, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
} 