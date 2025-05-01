/**
 * AI Models Data API Hooks
 *
 * This file contains React hooks for interacting with AI model-related API endpoints.
 * These hooks handle data fetching, caching, and real-time updates for AI model management.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import {
  getAIModelStats,
  getAIModels,
  getModelById,
  retrainModel,
  updateModelStatus,
  type AIModel,
  type AIModelStats,
} from "@/lib/api/ai-models"

/**
 * Hook for fetching and managing AI model statistics
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing stats data, loading state, error state, and refetch function
 *
 * @example
 * const { stats, loading, error } = useAIModelStats();
 */
export function useAIModelStats(refreshInterval = 30000) {
  const [stats, setStats] = useState<AIModelStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getAIModelStats()
      setStats(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch AI model stats:", err)
      setError("Failed to load AI model statistics")
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
 * Hook for fetching and managing AI models list
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing models array, loading state, error state, refetch function, and model management functions
 *
 * @example
 * const { models, loading, error, startRetraining } = useAIModels();
 */
export function useAIModels(refreshInterval = 30000) {
  const [models, setModels] = useState<AIModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getAIModels()
      setModels(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch AI models:", err)
      setError("Failed to load AI models")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchModels, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchModels, refreshInterval])

  /**
   * Start retraining an AI model
   *
   * @param id - The ID of the model to retrain
   * @returns Result object with success status and message
   */
  const startRetraining = useCallback(async (id: string) => {
    try {
      const result = await retrainModel(id)

      if (result.success) {
        // Update the model status in the local state
        setModels((prev) => prev.map((model) => (model.id === id ? { ...model, status: "Training" } : model)))
      }

      return result
    } catch (err) {
      console.error("Failed to retrain model:", err)
      throw err
    }
  }, [])

  /**
   * Update an AI model's status
   *
   * @param id - The ID of the model to update
   * @param status - The new status to set
   * @returns The updated model object
   */
  const updateStatus = useCallback(async (id: string, status: string) => {
    try {
      const updatedModel = await updateModelStatus(id, status)
      // Update the model in the local state
      setModels((prev) => prev.map((model) => (model.id === id ? updatedModel : model)))
      return updatedModel
    } catch (err) {
      console.error("Failed to update model status:", err)
      throw err
    }
  }, [])

  return {
    models,
    loading,
    error,
    refetch: fetchModels,
    startRetraining,
    updateStatus,
  }
}

/**
 * Hook for fetching details of a specific AI model
 *
 * @param id - The ID of the model to fetch
 * @returns Object containing model data, loading state, error state, and fetch function
 *
 * @example
 * const { model, loading, error } = useModelDetails("model-123");
 */
export function useModelDetails(id?: string) {
  const [model, setModel] = useState<AIModel | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchModel = useCallback(async (modelId: string) => {
    if (!modelId) return

    try {
      setLoading(true)
      const data = await getModelById(modelId)
      setModel(data)
      setError(null)
      return data
    } catch (err) {
      console.error("Failed to fetch model details:", err)
      setError("Failed to load model details")
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (id) {
      fetchModel(id)
    }
  }, [id, fetchModel])

  return { model, loading, error, fetchModel }
}
