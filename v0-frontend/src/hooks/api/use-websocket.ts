/**
 * WebSocket API Hooks
 *
 * This file contains React hooks for real-time communication using WebSockets.
 * These hooks handle connection management, message handling, and reconnection logic.
 */

"use client"

import { useState, useEffect, useCallback, useRef } from "react"

/**
 * Type definition for WebSocket messages
 */
type WebSocketMessage = {
  type: string
  data: any
}

/**
 * Type definition for WebSocket hook options
 */
type WebSocketOptions = {
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

/**
 * Hook for managing WebSocket connections
 *
 * @param url - WebSocket server URL
 * @param options - Configuration options for the WebSocket connection
 * @returns Object containing connection state, last message, error state, and WebSocket management functions
 *
 * @example
 * const { isConnected, lastMessage, sendMessage } = useWebSocket('wss://api.example.com/ws');
 */
export function useWebSocket(url: string, options: WebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [error, setError] = useState<Event | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectIntervalRef = useRef<number | null>(null)

  const { onOpen, onClose, onError, reconnectInterval = 5000, maxReconnectAttempts = 5 } = options

  /**
   * Establish a WebSocket connection
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        if (onOpen) onOpen()
      }

      ws.onclose = () => {
        setIsConnected(false)
        if (onClose) onClose()

        // Attempt to reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectIntervalRef.current = window.setTimeout(() => {
            reconnectAttemptsRef.current += 1
            connect()
          }, reconnectInterval)
        }
      }

      ws.onerror = (err) => {
        setError(err)
        if (onError) onError(err)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          setLastMessage(message)
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err)
        }
      }

      wsRef.current = ws
    } catch (err) {
      console.error("WebSocket connection error:", err)
      setError(err as Event)
    }
  }, [url, onOpen, onClose, onError, reconnectInterval, maxReconnectAttempts])

  /**
   * Close the WebSocket connection and clean up
   */
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    if (reconnectIntervalRef.current) {
      clearTimeout(reconnectIntervalRef.current)
      reconnectIntervalRef.current = null
    }
  }, [])

  /**
   * Send a message through the WebSocket connection
   *
   * @param data - Data to send
   * @returns true if message was sent successfully, false otherwise
   */
  const sendMessage = useCallback((data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
      return true
    }
    return false
  }, [])

  // Connect when the component mounts and disconnect when it unmounts
  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    connect,
    disconnect,
  }
}

/**
 * Hook for real-time updates from the NeuraShield system
 *
 * @returns Object containing connection state, error state, and categorized updates
 *
 * @example
 * const { isConnected, threatUpdates, blockchainUpdates } = useRealTimeUpdates();
 */
export function useRealTimeUpdates() {
  // Use environment variable for WebSocket URL or fallback to derived URL
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || `ws://${window.location.host}/api/ws`
  const { isConnected, lastMessage, error, sendMessage } = useWebSocket(wsUrl)

  const [threatUpdates, setThreatUpdates] = useState([])
  const [blockchainUpdates, setBlockchainUpdates] = useState([])
  const [systemAlerts, setSystemAlerts] = useState([])

  // Process incoming messages based on their type
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case "threat":
          setThreatUpdates((prev) => [lastMessage.data, ...prev].slice(0, 10))
          break
        case "blockchain":
          setBlockchainUpdates((prev) => [lastMessage.data, ...prev].slice(0, 10))
          break
        case "system":
          setSystemAlerts((prev) => [lastMessage.data, ...prev].slice(0, 10))
          break
        default:
          break
      }
    }
  }, [lastMessage])

  return {
    isConnected,
    error,
    threatUpdates,
    blockchainUpdates,
    systemAlerts,
    sendMessage,
  }
}
