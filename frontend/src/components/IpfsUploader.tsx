import React, { useRef, useState } from 'react';
import { useIpfs } from '@/lib/hooks/useIpfs';

interface IpfsUploaderProps {
  onUploadComplete?: (ipfsHash: string) => void;
}

export function IpfsUploader({ onUploadComplete }: IpfsUploaderProps) {
  const { uploadFile, uploadStatus } = useIpfs();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent<HTMLDivElement | HTMLFormElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement | HTMLFormElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (selectedFile) {
      const ipfsHash = await uploadFile(selectedFile);
      if (ipfsHash && onUploadComplete) {
        onUploadComplete(ipfsHash);
      }
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <form 
        className="mt-4"
        onSubmit={handleSubmit}
        onDragEnter={handleDrag}
      >
        <div 
          className={`p-6 border-2 border-dashed rounded-lg ${
            dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={handleChange}
          />
          
          <div className="text-center">
            <div className="mb-3">
              <svg 
                className="mx-auto h-12 w-12 text-gray-400" 
                stroke="currentColor" 
                fill="none" 
                viewBox="0 0 48 48" 
                aria-hidden="true"
              >
                <path 
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4h-8m-12 0H8m16 0a4 4 0 01-4-4v-4m0 0h-4m28 4h-4" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                />
              </svg>
            </div>
            
            <p className="text-sm text-gray-600 mb-1">
              Drag and drop file here, or
            </p>
            <button
              type="button"
              onClick={handleButtonClick}
              className="font-medium text-blue-600 hover:text-blue-500"
            >
              Select file
            </button>
            
            {selectedFile && (
              <div className="mt-3 py-2 px-3 bg-gray-100 rounded text-sm">
                {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
              </div>
            )}
          </div>
        </div>
        
        {uploadStatus.error && (
          <div className="mt-3 text-sm text-red-600">
            Error: {uploadStatus.error}
          </div>
        )}
        
        {uploadStatus.data && (
          <div className="mt-3 text-sm text-green-600">
            Successfully uploaded to IPFS with hash:
            <div className="font-mono text-xs break-all mt-1 p-2 bg-gray-50 rounded">
              {uploadStatus.data.ipfsHash}
            </div>
          </div>
        )}
        
        <div className="mt-4">
          <button
            type="submit"
            disabled={!selectedFile || uploadStatus.loading}
            className={`w-full py-2 px-4 rounded-md text-white font-medium ${
              !selectedFile || uploadStatus.loading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {uploadStatus.loading ? 'Uploading...' : 'Upload to IPFS'}
          </button>
        </div>
      </form>
    </div>
  );
} 