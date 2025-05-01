/**
 * Settings Data API Hook
 *
 * This file contains a React hook for interacting with system settings API endpoints.
 * This hook handles fetching, caching, and updating system settings.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import { getSettings, updateSettings, type SystemSettings } from "@/lib/api/settings"

/**
 * Hook for fetching and managing system settings
 *
 * @returns Object containing settings data, loading state, error state, saving state, and settings management functions
 *
 * @example
 * const { settings, loading, error, saving, saveSettings } = useSettings();
 * if (loading) return <LoadingSpinner />;
 * return <SettingsForm settings={settings} onSave={saveSettings} saving={saving} />;
 */
export function useSettings() {
  const [settings, setSettings] = useState<SystemSettings | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchSettings = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getSettings()
      setSettings(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch settings:", err)
      setError("Failed to load settings")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchSettings()
  }, [fetchSettings])

  /**
   * Save updated system settings
   *
   * @param updatedSettings - Partial settings object with fields to update
   * @returns The updated settings object
   */
  const saveSettings = useCallback(async (updatedSettings: Partial<SystemSettings>) => {
    try {
      setSaving(true)
      const result = await updateSettings(updatedSettings)
      setSettings(result)
      return result
    } catch (err) {
      console.error("Failed to save settings:", err)
      throw err
    } finally {
      setSaving(false)
    }
  }, [])

  return {
    settings,
    loading,
    error,
    saving,
    fetchSettings,
    saveSettings,
  }
}
