import React, { useState } from 'react';
import LoadingSpinner from '../components/ui/loading-spinner';
import Alert from '../components/ui/alert';

const ModelTraining: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);
  const [formData, setFormData] = useState({
    epochs: 100,
    batchSize: 64,
    learningRate: 0.001,
    datasetSize: 10000,
    validationSplit: 0.2
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: parseFloat(value)
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsTraining(true);
    setResult(null);

    try {
      // Call the backend API to start model training
      const response = await fetch('/api/train-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.success) {
        // If training started successfully, poll for status
        const jobId = data.jobId;
        
        // Poll for training status every 2 seconds
        const interval = setInterval(async () => {
          try {
            const statusResponse = await fetch(`/api/train-model/${jobId}`);
            
            if (!statusResponse.ok) {
              throw new Error(`Error fetching status: ${statusResponse.statusText}`);
            }
            
            const statusData = await statusResponse.json();
            
            // If training is completed or failed, stop polling
            if (statusData.status === 'completed' || statusData.status === 'failed') {
              clearInterval(interval);
              setIsTraining(false);
              
              setResult({
                success: statusData.status === 'completed',
                message: statusData.message
              });
            }
          } catch (error: any) {
            console.error('Error polling training status:', error);
            clearInterval(interval);
            setIsTraining(false);
            setResult({
              success: false,
              message: `Error monitoring training status: ${error.message}`
            });
          }
        }, 2000);
        
        // Timeout after 5 minutes to avoid infinite polling
        setTimeout(() => {
          clearInterval(interval);
          if (isTraining) {
            setIsTraining(false);
            setResult({
              success: false,
              message: 'Training timeout. The job may still be running in the background.'
            });
          }
        }, 5 * 60 * 1000);
        
      } else {
        setIsTraining(false);
        setResult({
          success: false,
          message: data.message || 'Unknown error occurred'
        });
      }
    } catch (error: any) {
      console.error('Error starting model training:', error);
      setIsTraining(false);
      setResult({
        success: false,
        message: `Error starting model training: ${error.message}`
      });
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Model Training</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold mb-4">Training Parameters</h2>
          
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Epochs
                </label>
                <input
                  type="number"
                  name="epochs"
                  value={formData.epochs}
                  onChange={handleChange}
                  min="1"
                  max="1000"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  name="batchSize"
                  value={formData.batchSize}
                  onChange={handleChange}
                  min="1"
                  max="512"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Learning Rate
                </label>
                <input
                  type="number"
                  name="learningRate"
                  value={formData.learningRate}
                  onChange={handleChange}
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dataset Size
                </label>
                <input
                  type="number"
                  name="datasetSize"
                  value={formData.datasetSize}
                  onChange={handleChange}
                  min="1000"
                  max="100000"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Validation Split
                </label>
                <input
                  type="number"
                  name="validationSplit"
                  value={formData.validationSplit}
                  onChange={handleChange}
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  required
                />
              </div>
              
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isTraining}
                  className="w-full btn btn-primary flex justify-center items-center"
                >
                  {isTraining ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Training...
                    </>
                  ) : (
                    'Start Training'
                  )}
                </button>
              </div>
            </div>
          </form>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold mb-4">Training Status</h2>
          
          {isTraining ? (
            <div className="flex flex-col items-center justify-center space-y-4 p-8">
              <LoadingSpinner size="lg" />
              <p className="text-gray-600">Training in progress...</p>
            </div>
          ) : result ? (
            <div className="space-y-4">
              <Alert 
                type={result.success ? 'success' : 'error'} 
                message={result.message}
              />
              
              {result.success && (
                <div className="mt-6 space-y-4">
                  <h3 className="text-lg font-medium">Training Results</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Accuracy</div>
                      <div className="text-xl font-semibold">95.2%</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Loss</div>
                      <div className="text-xl font-semibold">0.164</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Training Time</div>
                      <div className="text-xl font-semibold">4m 32s</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Model Size</div>
                      <div className="text-xl font-semibold">2.3 MB</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500 text-center p-8">
              Configure parameters and start training to see results
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelTraining; 