import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/ui/loading-spinner';
import Alert from '../components/ui/alert';

interface Event {
  id: string;
  timestamp: string;
  type: string;
  details: Record<string, any>;
  ipfsHash: string;
}

const Events: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/events');
        
        if (!response.ok) {
          throw new Error(`Error fetching events: ${response.statusText}`);
        }
        
        const data = await response.json();
        setEvents(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching events:', err);
        setError('Failed to load security events. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchEvents();
  }, []);

  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleString();
  };

  const getEventTypeColor = (type: string): string => {
    switch (type.toLowerCase()) {
      case 'access_violation':
        return 'bg-red-100 text-red-800';
      case 'data_breach':
        return 'bg-purple-100 text-purple-800';
      case 'malware_detected':
        return 'bg-orange-100 text-orange-800';
      case 'network_scan':
        return 'bg-yellow-100 text-yellow-800';
      case 'authentication_failure':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Security Events</h1>
      
      {error && (
        <Alert type="error" message={error} className="mb-4" />
      )}
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 bg-white rounded-lg shadow-sm overflow-hidden">
            <div className="border-b px-4 py-3 bg-gray-50">
              <h2 className="font-medium">Event List</h2>
            </div>
            <div className="overflow-y-auto" style={{ maxHeight: '600px' }}>
              {events.length === 0 ? (
                <div className="p-4 text-gray-500">No events found</div>
              ) : (
                <ul className="divide-y">
                  {events.map((event) => (
                    <li 
                      key={event.id}
                      className={`p-4 cursor-pointer hover:bg-gray-50 ${selectedEvent?.id === event.id ? 'bg-gray-100' : ''}`}
                      onClick={() => setSelectedEvent(event)}
                    >
                      <div className="flex justify-between">
                        <span className={`px-2 py-1 rounded-full text-xs ${getEventTypeColor(event.type)}`}>
                          {event.type}
                        </span>
                        <span className="text-xs text-gray-500">{formatTimestamp(event.timestamp)}</span>
                      </div>
                      <div className="mt-2 text-sm font-medium truncate">{event.id}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          
          <div className="lg:col-span-2 bg-white rounded-lg shadow-sm">
            {selectedEvent ? (
              <div className="p-6">
                <div className="mb-6">
                  <h2 className="text-xl font-semibold mb-2">{selectedEvent.type}</h2>
                  <div className="text-sm text-gray-500 mb-2">
                    {formatTimestamp(selectedEvent.timestamp)}
                  </div>
                  <div className="text-sm mb-2">
                    <span className="font-medium">Event ID:</span> {selectedEvent.id}
                  </div>
                </div>
                
                <div className="mb-6">
                  <h3 className="text-lg font-medium mb-2">Details</h3>
                  <pre className="bg-gray-50 p-4 rounded overflow-auto text-sm">
                    {JSON.stringify(selectedEvent.details, null, 2)}
                  </pre>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Blockchain Record</h3>
                  <div className="bg-gray-50 p-4 rounded overflow-auto text-sm font-mono">
                    <div className="mb-2">
                      <span className="font-medium text-gray-600">IPFS Hash: </span>
                      <span className="text-primary">{selectedEvent.ipfsHash}</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex justify-center items-center h-full p-6 text-gray-500">
                Select an event to view details
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Events; 