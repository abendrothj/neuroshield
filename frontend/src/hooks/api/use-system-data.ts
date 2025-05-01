/**
 * System Data API Hooks
 *
 * This file contains React hooks for interacting with system-related API endpoints.
 * These hooks handle data fetching, caching, and real-time updates for system health and status.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import {
  getSystemHealth,
  getBlockchainStatus,
  getIPFSStatus,
  type SystemHealth,
  type BlockchainStatus,
  type IPFSStatus,
} from "@/lib/api/system"

/**
 * Hook for fetching and managing system health data
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing health data, loading state, error state, and refetch function
 *
 * @example
 * const { health, loading, error } = useSystemHealth();
 */
export function useSystemHealth(refreshInterval = 30000) {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchHealth = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getSystemHealth()
      setHealth(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch system health:", err)
      setError("Failed to load system health data")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchHealth, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchHealth, refreshInterval])

  return { health, loading, error, refetch: fetchHealth }
}

/**
 * Hook for fetching and managing blockchain status data
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing status data, loading state, error state, and refetch function
 *
 * @example
 * const { status, loading, error } = useBlockchainStatus();
 */
export function useBlockchainStatus(refreshInterval = 30000) {
  const [status, setStatus] = useState<BlockchainStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getBlockchainStatus()
      setStatus(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch blockchain status:", err)
      setError("Failed to load blockchain status")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchStatus, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchStatus, refreshInterval])

  return { status, loading, error, refetch: fetchStatus }
}

/**
 * Hook for fetching and managing IPFS status data
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing status data, loading state, error state, and refetch function
 *
 * @example
 * const { status, loading, error } = useIPFSStatus();
 */
export function useIPFSStatus(refreshInterval = 30000) {
  const [status, setStatus] = useState<IPFSStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getIPFSStatus()
      setStatus(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch IPFS status:", err)
      setError("Failed to load IPFS status")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchStatus, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchStatus, refreshInterval])

  return { status, loading, error, refetch: fetchStatus }
}
