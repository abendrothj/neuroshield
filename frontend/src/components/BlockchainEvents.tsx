import React, { useState, useMemo } from 'react';
import { useBlockchain } from '@/lib/hooks/useBlockchain';
import { SecurityEvent, EventType } from '@/lib/types/blockchain';

const severityColors = {
  LOW: 'bg-blue-100 text-blue-800',
  MEDIUM: 'bg-yellow-100 text-yellow-800',
  HIGH: 'bg-orange-100 text-orange-800',
  CRITICAL: 'bg-red-100 text-red-800',
};

const statusColors = {
  PENDING: 'bg-gray-100 text-gray-800',
  CONFIRMED: 'bg-green-100 text-green-800',
  FAILED: 'bg-red-100 text-red-800',
};

type SortField = 'timestamp' | 'severity' | 'type';
type SortDirection = 'asc' | 'desc';

export function BlockchainEvents() {
  const { events, fetchSecurityEvents, eventTypes } = useBlockchain();
  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');
  const [typeFilter, setTypeFilter] = useState<string>('ALL');
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

  const filteredAndSortedEvents = useMemo(() => {
    if (!events.data) return [];

    // Filter by search term, severity, and type
    let filtered = events.data.filter((event) => {
      const matchesSearch = searchTerm === '' || 
        event.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        event.source.toLowerCase().includes(searchTerm.toLowerCase());
        
      const matchesSeverity = severityFilter === 'ALL' || event.severity === severityFilter;
      const matchesType = typeFilter === 'ALL' || event.type === typeFilter;
      
      return matchesSearch && matchesSeverity && matchesType;
    });

    // Sort the filtered events
    return filtered.sort((a, b) => {
      let comparison = 0;
      
      if (sortField === 'timestamp') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortField === 'severity') {
        const severityOrder = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
        comparison = severityOrder[a.severity] - severityOrder[b.severity];
      } else if (sortField === 'type') {
        comparison = a.type.localeCompare(b.type);
      }
      
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [events.data, searchTerm, severityFilter, typeFilter, sortField, sortDirection]);

  if (events.loading) {
    return <div className="p-4">Loading events...</div>;
  }

  if (events.error) {
    return (
      <div className="p-4 text-red-600">
        Error: {events.error}
        <button
          onClick={fetchSecurityEvents}
          className="ml-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!events.data || events.data.length === 0) {
    return <div className="p-4">No events found.</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <h2 className="text-2xl font-bold">Security Events</h2>
        <div className="flex flex-col md:flex-row gap-2 w-full md:w-auto">
          <input
            type="text"
            placeholder="Search events..."
            className="px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          
          <select
            className="px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
          >
            <option value="ALL">All Severities</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
            <option value="CRITICAL">Critical</option>
          </select>
          
          <select
            className="px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
          >
            <option value="ALL">All Types</option>
            {Object.values(EventType).map((type) => (
              <option key={type} value={type}>
                {type.replace('_', ' ')}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="overflow-x-auto bg-white rounded-lg shadow">
        <div className="flex items-center justify-between p-4 border-b">
          <div className="text-sm text-gray-500">
            Showing {filteredAndSortedEvents.length} of {events.data.length} events
          </div>
          <div className="flex gap-2">
            <button 
              onClick={() => handleSort('timestamp')}
              className={`px-2 py-1 text-xs rounded ${sortField === 'timestamp' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}
            >
              Time {sortField === 'timestamp' && (sortDirection === 'asc' ? '↑' : '↓')}
            </button>
            <button 
              onClick={() => handleSort('severity')}
              className={`px-2 py-1 text-xs rounded ${sortField === 'severity' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}
            >
              Severity {sortField === 'severity' && (sortDirection === 'asc' ? '↑' : '↓')}
            </button>
            <button 
              onClick={() => handleSort('type')}
              className={`px-2 py-1 text-xs rounded ${sortField === 'type' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}
            >
              Type {sortField === 'type' && (sortDirection === 'asc' ? '↑' : '↓')}
            </button>
          </div>
        </div>
        
        <div className="grid gap-4 p-4">
          {filteredAndSortedEvents.map((event: SecurityEvent) => (
            <div
              key={event.id}
              className="border rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold">{event.description}</h3>
                <div className="flex gap-2">
                  <span className={`px-2 py-1 rounded text-sm ${severityColors[event.severity]}`}>
                    {event.severity}
                  </span>
                  <span className={`px-2 py-1 rounded text-sm ${statusColors[event.status]}`}>
                    {event.status}
                  </span>
                </div>
              </div>
              <div className="text-sm text-gray-600">
                <p>Type: {event.type}</p>
                <p>Source: {event.source}</p>
                <p>Time: {new Date(event.timestamp).toLocaleString()}</p>
                {event.blockchainTxHash && (
                  <p className="font-mono text-xs mt-2">
                    TX: {event.blockchainTxHash}
                  </p>
                )}
              </div>
              {event.metadata && (
                <div className="mt-2 p-2 bg-gray-50 rounded">
                  <p className="text-sm font-semibold">Metadata:</p>
                  <pre className="text-xs overflow-auto">
                    {JSON.stringify(event.metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 