"use client"

import { useState, useEffect, useCallback } from "react"
import { api, type ThreatData, type BlockchainLog } from "@/lib/api"

export function useThreatDetection() {
  const [threatData, setThreatData] = useState<ThreatData[]>([])
  const [blockchainLogs, setBlockchainLogs] = useState<BlockchainLog[]>([])
  const [autoResponseEnabled, setAutoResponseEnabled] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [threats, logs] = await Promise.all([api.getThreats(), api.getLogs()])
        setThreatData([threats])
        setBlockchainLogs(logs)
      } catch (err) {
        setError("Failed to fetch initial data")
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchInitialData()
  }, [])

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = api.connectWebSocket((data: ThreatData) => {
      setThreatData((prev) => [...prev, data].slice(-50)) // Keep last 50 data points
    })

    return () => {
      ws.close()
    }
  }, [])

  // Auto-response mode
  useEffect(() => {
    if (autoResponseEnabled) {
      const interval = setInterval(async () => {
        try {
          const logs = await api.getLogs()
          setBlockchainLogs(logs)
        } catch (err) {
          console.error("Failed to fetch logs:", err)
        }
      }, 10000)

      return () => clearInterval(interval)
    }
  }, [autoResponseEnabled])

  // Toggle auto-response mode
  const toggleAutoResponse = useCallback(async () => {
    try {
      const { autoResponseEnabled: newState } = await api.toggleAutoResponse()
      setAutoResponseEnabled(newState)
    } catch (err) {
      setError("Failed to toggle auto-response mode")
      console.error(err)
    }
  }, [])

  // Trigger manual response
  const triggerResponse = useCallback(async (ip: string) => {
    try {
      await api.triggerResponse(ip)
      const logs = await api.getLogs()
      setBlockchainLogs(logs)
    } catch (err) {
      setError("Failed to trigger response")
      console.error(err)
    }
  }, [])

  // Refresh logs
  const refreshLogs = useCallback(async () => {
    try {
      const logs = await api.getLogs()
      setBlockchainLogs(logs)
    } catch (err) {
      setError("Failed to refresh logs")
      console.error(err)
    }
  }, [])

  return {
    threatData,
    blockchainLogs,
    autoResponseEnabled,
    isLoading,
    error,
    toggleAutoResponse,
    triggerResponse,
    refreshLogs,
  }
}

