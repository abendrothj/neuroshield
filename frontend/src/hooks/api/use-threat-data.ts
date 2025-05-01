/**
 * Threat Data API Hooks
 *
 * This file contains React hooks for interacting with the threat-related API endpoints.
 * These hooks handle data fetching, caching, and real-time updates for threat management.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import {
  getThreatStats,
  getThreats,
  getThreatById,
  updateThreatStatus,
  getThreatTrends,
  type Threat,
  type ThreatStats,
  type ThreatTrend,
} from "@/lib/api/threats"

/**
 * Hook for fetching and managing threat statistics
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing stats data, loading state, error state, and refetch function
 *
 * @example
 * const { stats, loading, error, refetch } = useThreatStats();
 * if (loading) return <LoadingSpinner />;
 * if (error) return <ErrorMessage message={error} />;
 * return <ThreatStatsDisplay stats={stats} />;
 */
export function useThreatStats(refreshInterval = 30000) {
  const [stats, setStats] = useState<ThreatStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getThreatStats()
      setStats(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch threat stats:", err)
      setError("Failed to load threat statistics")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStats()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchStats, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchStats, refreshInterval])

  return { stats, loading, error, refetch: fetchStats }
}

/**
 * Hook for fetching and managing threat list data with filtering options
 *
 * @param severity - Filter threats by severity level
 * @param status - Filter threats by status
 * @param searchQuery - Search query to filter threats
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 10000ms)
 * @returns Object containing threats array, loading state, error state, refetch function, and update function
 *
 * @example
 * const { threats, loading, error, updateThreat } = useThreats("High", "Active", "network");
 */
export function useThreats(severity?: string, status?: string, searchQuery?: string, refreshInterval = 10000) {
  const [threats, setThreats] = useState<Threat[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchThreats = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getThreats(severity, status, searchQuery)
      setThreats(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch threats:", err)
      setError("Failed to load threats")
    } finally {
      setLoading(false)
    }
  }, [severity, status, searchQuery])

  useEffect(() => {
    fetchThreats()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchThreats, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchThreats, refreshInterval])

  /**
   * Update a threat's status
   *
   * @param id - The ID of the threat to update
   * @param newStatus - The new status to set
   * @returns The updated threat object
   */
  const updateThreat = useCallback(async (id: string, newStatus: string) => {
    try {
      const updatedThreat = await updateThreatStatus(id, newStatus)
      // Update the threat in the local state
      setThreats((prev) => prev.map((threat) => (threat.id === id ? updatedThreat : threat)))
      return updatedThreat
    } catch (err) {
      console.error("Failed to update threat status:", err)
      throw err
    }
  }, [])

  return {
    threats,
    loading,
    error,
    refetch: fetchThreats,
    updateThreat,
  }
}

/**
 * Hook for fetching and managing details of a specific threat
 *
 * @param id - The ID of the threat to fetch
 * @returns Object containing threat data, loading state, error state, and fetch function
 *
 * @example
 * const { threat, loading, error } = useThreatDetails("THR-1234");
 */
export function useThreatDetails(id?: string) {
  const [threat, setThreat] = useState<Threat | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchThreat = useCallback(async (threatId: string) => {
    if (!threatId) return

    try {
      setLoading(true)
      const data = await getThreatById(threatId)
      setThreat(data)
      setError(null)
      return data
    } catch (err) {
      console.error("Failed to fetch threat details:", err)
      setError("Failed to load threat details")
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (id) {
      fetchThreat(id)
    }
  }, [id, fetchThreat])

  return { threat, loading, error, fetchThreat }
}

/**
 * Hook for fetching threat trend data over time
 *
 * @param days - Number of days to fetch trend data for (default: 7)
 * @returns Object containing trends array, loading state, error state, and refetch function
 *
 * @example
 * const { trends, loading, error } = useThreatTrends(30); // Get trends for last 30 days
 */
export function useThreatTrends(days = 7) {
  const [trends, setTrends] = useState<ThreatTrend[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTrends = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getThreatTrends(days)
      setTrends(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch threat trends:", err)
      setError("Failed to load threat trend data")
    } finally {
      setLoading(false)
    }
  }, [days])

  useEffect(() => {
    fetchTrends()
  }, [fetchTrends])

  return { trends, loading, error, refetch: fetchTrends }
}
