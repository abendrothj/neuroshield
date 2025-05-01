import React, { useState } from 'react';
import { useBlockchain } from '@/lib/hooks/useBlockchain';
import { IpfsViewer } from './IpfsViewer';
import { IpfsUploader } from './IpfsUploader';
import { SecurityEvent } from '@/lib/types/blockchain';

export function Evidence() {
  const { events } = useBlockchain();
  const [selectedEvent, setSelectedEvent] = useState<SecurityEvent | null>(null);
  const [showUploader, setShowUploader] = useState(false);

  // Filter events that have blockchainTxHash (proof they're on the blockchain)
  const verifiedEvents = events.data?.filter(event => 
    event.blockchainTxHash && event.blockchainTxHash.length > 0
  ) || [];

  const handleEventSelect = (event: SecurityEvent) => {
    setSelectedEvent(event);
  };

  const handleUploadComplete = (ipfsHash: string) => {
    // In a real implementation, we would attach this hash to a security event
    console.log(`File uploaded with IPFS hash: ${ipfsHash}`);
    setShowUploader(false);
  };

  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col md:flex-row justify-between items-start mb-6">
        <h2 className="text-2xl font-bold">Forensic Evidence</h2>
        <button
          onClick={() => setShowUploader(!showUploader)}
          className="px-4 py-2 mt-2 md:mt-0 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {showUploader ? 'Cancel Upload' : 'Upload Evidence'}
        </button>
      </div>

      {showUploader ? (
        <div className="bg-white p-6 rounded-lg shadow mb-6">
          <h3 className="text-lg font-semibold mb-4">Upload Evidence to IPFS</h3>
          <IpfsUploader onUploadComplete={handleUploadComplete} />
        </div>
      ) : (
        <>
          {selectedEvent ? (
            <div className="bg-white p-6 rounded-lg shadow mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Event Evidence</h3>
                <button
                  onClick={() => setSelectedEvent(null)}
                  className="text-sm text-gray-600 hover:text-gray-800"
                >
                  Back to Events
                </button>
              </div>
              
              <div className="mb-4 p-4 bg-gray-50 rounded">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="font-medium">Type:</span> {selectedEvent.type}
                  </div>
                  <div>
                    <span className="font-medium">Severity:</span> {selectedEvent.severity}
                  </div>
                  <div className="col-span-2">
                    <span className="font-medium">Description:</span> {selectedEvent.description}
                  </div>
                  <div className="col-span-2">
                    <span className="font-medium">Timestamp:</span> {new Date(selectedEvent.timestamp).toLocaleString()}
                  </div>
                  <div className="col-span-2">
                    <span className="font-medium">Blockchain TX:</span> {selectedEvent.blockchainTxHash}
                  </div>
                </div>
              </div>
              
              {selectedEvent.metadata?.ipfsHash ? (
                <IpfsViewer ipfsHash={selectedEvent.metadata.ipfsHash} />
              ) : (
                <div className="text-gray-500 text-center p-4 border rounded">
                  No IPFS evidence attached to this event.
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="font-medium">Verified Security Events</h3>
                <p className="text-sm text-gray-500">
                  Security events with blockchain verification. Select an event to view evidence.
                </p>
              </div>
              {verifiedEvents.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  No verified events found. Events will appear here once they have been recorded on the blockchain.
                </div>
              ) : (
                <div className="divide-y">
                  {verifiedEvents.map(event => (
                    <div
                      key={event.id}
                      className="p-4 hover:bg-gray-50 cursor-pointer"
                      onClick={() => handleEventSelect(event)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-medium">{event.type}</span>
                          <p className="text-sm text-gray-600">{event.description}</p>
                        </div>
                        <div className="flex gap-2 items-center">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            event.severity === 'CRITICAL' ? 'bg-red-100 text-red-800' :
                            event.severity === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                            event.severity === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-blue-100 text-blue-800'
                          }`}>
                            {event.severity}
                          </span>
                          <span className="text-xs text-gray-500">
                            {new Date(event.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        <span className="font-mono">{event.blockchainTxHash?.substring(0, 20)}...</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
} 