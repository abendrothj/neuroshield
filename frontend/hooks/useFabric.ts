import { useEffect, useState } from 'react';
import FabricService from '@/lib/fabricService';

export function useFabric() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const fabricService = FabricService.getInstance();

  useEffect(() => {
    const initializeFabric = async () => {
      try {
        await fabricService.initialize();
        setIsInitialized(true);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to initialize Fabric'));
      }
    };

    initializeFabric();

    return () => {
      fabricService.disconnect();
    };
  }, []);

  const recordThreat = async (threatData: {
    type: string;
    severity: string;
    timestamp: string;
    details: string;
  }) => {
    try {
      return await fabricService.recordThreat(threatData);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to record threat'));
      throw err;
    }
  };

  const getThreatHistory = async () => {
    try {
      return await fabricService.getThreatHistory();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to get threat history'));
      throw err;
    }
  };

  const getSystemMetrics = async () => {
    try {
      return await fabricService.getSystemMetrics();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to get system metrics'));
      throw err;
    }
  };

  const updateSystemMetrics = async (metrics: {
    cpu: number;
    memory: number;
    network: number;
    timestamp: string;
  }) => {
    try {
      return await fabricService.updateSystemMetrics(metrics);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to update system metrics'));
      throw err;
    }
  };

  return {
    isInitialized,
    error,
    recordThreat,
    getThreatHistory,
    getSystemMetrics,
    updateSystemMetrics,
  };
} 