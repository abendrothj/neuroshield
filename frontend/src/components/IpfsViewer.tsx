import React, { useEffect, useState } from 'react';
import { useIpfs } from '@/lib/hooks/useIpfs';

interface IpfsViewerProps {
  ipfsHash: string;
}

export function IpfsViewer({ ipfsHash }: IpfsViewerProps) {
  const { fileContent, fileMetadata, fetchFileContent, fetchFileMetadata, getGatewayUrl } = useIpfs();
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (ipfsHash) {
      fetchFileMetadata(ipfsHash);
    }
  }, [ipfsHash, fetchFileMetadata]);

  const handleViewContent = () => {
    fetchFileContent(ipfsHash);
    setIsExpanded(true);
  };

  const handleDownload = () => {
    if (!fileContent.data) {
      fetchFileContent(ipfsHash);
      return;
    }

    const blob = new Blob([fileContent.data.content], { type: fileContent.data.mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileContent.data.filename || `ipfs-${ipfsHash}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (fileMetadata.loading) {
    return <div className="p-4 text-center">Loading file metadata...</div>;
  }

  if (fileMetadata.error) {
    return (
      <div className="p-4 text-red-600">
        Error: {fileMetadata.error}
        <button
          onClick={() => fetchFileMetadata(ipfsHash)}
          className="ml-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!fileMetadata.data) {
    return <div className="p-4">No file data available for hash: {ipfsHash}</div>;
  }

  const isJsonContent = fileMetadata.data.mimeType === 'application/json';
  const isTextContent = fileMetadata.data.mimeType.includes('text/') || isJsonContent;
  const isImageContent = fileMetadata.data.mimeType.includes('image/');

  return (
    <div className="border rounded-lg p-4 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{fileMetadata.data.filename || `File: ${ipfsHash.substring(0, 10)}...`}</h3>
        <div className="flex space-x-2">
          <button
            onClick={handleViewContent}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
            disabled={fileContent.loading}
          >
            {fileContent.loading ? 'Loading...' : 'View'}
          </button>
          <button
            onClick={handleDownload}
            className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700"
          >
            Download
          </button>
        </div>
      </div>

      <div className="text-sm text-gray-600 mb-4">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="font-medium">Type:</span> {fileMetadata.data.mimeType}
          </div>
          <div>
            <span className="font-medium">Size:</span> {fileMetadata.data.size} bytes
          </div>
          <div className="col-span-2">
            <span className="font-medium">IPFS Hash:</span> {fileMetadata.data.hash}
          </div>
          <div className="col-span-2">
            <span className="font-medium">Timestamp:</span> {new Date(fileMetadata.data.timestamp).toLocaleString()}
          </div>
        </div>
      </div>

      {isExpanded && fileContent.loading && (
        <div className="p-4 text-center">Loading content...</div>
      )}

      {isExpanded && fileContent.error && (
        <div className="p-4 text-red-600">
          Error loading content: {fileContent.error}
          <button
            onClick={() => fetchFileContent(ipfsHash)}
            className="ml-4 px-2 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      )}

      {isExpanded && fileContent.data && (
        <div className="mt-4">
          <div className="border-t pt-4">
            <h4 className="text-md font-medium mb-2">Content:</h4>
            {isTextContent ? (
              <pre className="bg-gray-50 p-4 rounded text-sm overflow-auto max-h-80 whitespace-pre-wrap">
                {isJsonContent 
                  ? JSON.stringify(JSON.parse(fileContent.data.content), null, 2)
                  : fileContent.data.content
                }
              </pre>
            ) : isImageContent ? (
              <div className="flex justify-center">
                <img 
                  src={getGatewayUrl(ipfsHash)} 
                  alt={fileMetadata.data.filename || "IPFS content"} 
                  className="max-w-full max-h-96 object-contain"
                />
              </div>
            ) : (
              <div className="p-4 bg-gray-50 text-center">
                <p>Binary content (not displayed)</p>
                <a 
                  href={getGatewayUrl(ipfsHash)} 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  View in IPFS Gateway
                </a>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
} 