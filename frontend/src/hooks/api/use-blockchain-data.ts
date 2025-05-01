/**
 * Blockchain Data API Hooks
 *
 * This file contains React hooks for interacting with blockchain-related API endpoints.
 * These hooks handle data fetching, caching, and real-time updates for blockchain transactions and status.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import {
  getBlockchainTransactions,
  getBlockchainStats,
  getTransactionById,
  createTransaction,
  downloadTransactionsJson,
  type BlockchainTransaction,
  type BlockchainStats,
} from "@/lib/api/blockchain"

/**
 * Hook for fetching and managing blockchain transactions
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 10000ms)
 * @returns Object containing transactions array, loading state, error state, refetch function, and utility functions
 *
 * @example
 * const { transactions, loading, error, downloadTransactions } = useBlockchainTransactions();
 */
export function useBlockchainTransactions(refreshInterval = 10000) {
  const [transactions, setTransactions] = useState<BlockchainTransaction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTransactions = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getBlockchainTransactions()
      setTransactions(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch blockchain transactions:", err)
      setError("Failed to load blockchain transactions")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchTransactions()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchTransactions, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchTransactions, refreshInterval])

  /**
   * Add a new blockchain transaction
   *
   * @param data - Transaction data to create
   * @returns The newly created transaction
   */
  const addTransaction = useCallback(async (data: Omit<BlockchainTransaction, "id" | "status" | "block">) => {
    try {
      const newTransaction = await createTransaction(data)
      // Add the new transaction to the local state
      setTransactions((prev) => [newTransaction, ...prev])
      return newTransaction
    } catch (err) {
      console.error("Failed to create transaction:", err)
      throw err
    }
  }, [])

  /**
   * Download transactions as JSON file
   *
   * @returns true if download started successfully
   */
  const downloadTransactions = useCallback(() => {
    try {
      downloadTransactionsJson()
      return true
    } catch (err) {
      console.error("Failed to download transactions:", err)
      throw err
    }
  }, [])

  return {
    transactions,
    loading,
    error,
    refetch: fetchTransactions,
    addTransaction,
    downloadTransactions,
  }
}

/**
 * Hook for fetching and managing blockchain network statistics
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing stats data, loading state, error state, and refetch function
 *
 * @example
 * const { stats, loading, error } = useBlockchainStats();
 */
export function useBlockchainStats(refreshInterval = 30000) {
  const [stats, setStats] = useState<BlockchainStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getBlockchainStats()
      setStats(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch blockchain stats:", err)
      setError("Failed to load blockchain statistics")
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
 * Hook for fetching details of a specific blockchain transaction
 *
 * @param id - The ID of the transaction to fetch
 * @returns Object containing transaction data, loading state, error state, and fetch function
 *
 * @example
 * const { transaction, loading, error } = useTransactionDetails("0x1234...");
 */
export function useTransactionDetails(id?: string) {
  const [transaction, setTransaction] = useState<BlockchainTransaction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchTransaction = useCallback(async (txId: string) => {
    if (!txId) return

    try {
      setLoading(true)
      const data = await getTransactionById(txId)
      setTransaction(data)
      setError(null)
      return data
    } catch (err) {
      console.error("Failed to fetch transaction details:", err)
      setError("Failed to load transaction details")
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (id) {
      fetchTransaction(id)
    }
  }, [id, fetchTransaction])

  return { transaction, loading, error, fetchTransaction }
}
