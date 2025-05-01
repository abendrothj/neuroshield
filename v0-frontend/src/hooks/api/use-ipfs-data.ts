/**
 * IPFS Data API Hooks
 *
 * This file contains React hooks for interacting with IPFS-related API endpoints.
 * These hooks handle data fetching, caching, file uploads/downloads, and real-time updates for IPFS file management.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import {
  getIPFSFiles,
  getIPFSStats,
  getFileById,
  uploadFile,
  downloadFile,
  type IPFSFile,
  type IPFSStats,
} from "@/lib/api/ipfs"

/**
 * Hook for fetching and managing IPFS network statistics
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing stats data, loading state, error state, and refetch function
 *
 * @example
 * const { stats, loading, error } = useIPFSStats();
 */
export function useIPFSStats(refreshInterval = 30000) {
  const [stats, setStats] = useState<IPFSStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getIPFSStats()
      setStats(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch IPFS stats:", err)
      setError("Failed to load IPFS statistics")
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
 * Hook for fetching and managing IPFS files
 *
 * @param refreshInterval - Interval in milliseconds to refresh data (default: 30000ms)
 * @returns Object containing files array, loading state, error state, refetch function, and file management functions
 *
 * @example
 * const { files, loading, error, uploadFile, downloadFile } = useIPFSFiles();
 */
export function useIPFSFiles(refreshInterval = 30000) {
  const [files, setFiles] = useState<IPFSFile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchFiles = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getIPFSFiles()
      setFiles(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch IPFS files:", err)
      setError("Failed to load IPFS files")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchFiles()

    // Set up automatic refresh if interval is positive
    if (refreshInterval > 0) {
      const interval = setInterval(fetchFiles, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchFiles, refreshInterval])

  /**
   * Upload a new file to IPFS
   *
   * @param file - The file to upload
   * @param onProgress - Optional callback for upload progress updates
   * @returns The uploaded file metadata
   */
  const uploadNewFile = useCallback(async (file: File, onProgress?: (progress: number) => void) => {
    try {
      const uploadedFile = await uploadFile(file, onProgress)
      // Add the new file to the local state
      setFiles((prev) => [uploadedFile, ...prev])
      return uploadedFile
    } catch (err) {
      console.error("Failed to upload file:", err)
      throw err
    }
  }, [])

  /**
   * Download a file from IPFS
   *
   * @param cid - The content identifier of the file to download
   * @param filename - The name to save the file as
   * @returns true if download started successfully
   */
  const downloadIPFSFile = useCallback((cid: string, filename: string) => {
    try {
      downloadFile(cid, filename)
      return true
    } catch (err) {
      console.error("Failed to download file:", err)
      throw err
    }
  }, [])

  return {
    files,
    loading,
    error,
    refetch: fetchFiles,
    uploadFile: uploadNewFile,
    downloadFile: downloadIPFSFile,
  }
}

/**
 * Hook for fetching details of a specific IPFS file
 *
 * @param cid - The content identifier of the file to fetch
 * @returns Object containing file data, loading state, error state, and fetch function
 *
 * @example
 * const { file, loading, error } = useFileDetails("Qm123...");
 */
export function useFileDetails(cid?: string) {
  const [file, setFile] = useState<IPFSFile | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchFile = useCallback(async (fileCid: string) => {
    if (!fileCid) return

    try {
      setLoading(true)
      const data = await getFileById(fileCid)
      setFile(data)
      setError(null)
      return data
    } catch (err) {
      console.error("Failed to fetch file details:", err)
      setError("Failed to load file details")
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (cid) {
      fetchFile(cid)
    }
  }, [cid, fetchFile])

  return { file, loading, error, fetchFile }
}
