"use client"

import { useState, useCallback } from "react"
import { useToast } from "@/hooks/use-toast"

export type ApiError = {
  message: string
  statusCode?: number
  path?: string
  timestamp?: string
  retryable?: boolean
}

export function useApiError() {
  const [error, setError] = useState<ApiError | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const { toast } = useToast()

  // Function to handle API errors
  const handleError = useCallback(
    (err: unknown, customMessage?: string) => {
      let errorMessage = customMessage || "An error occurred"
      let statusCode: number | undefined = undefined
      let retryable = false

      if (err instanceof Error) {
        errorMessage = err.message
      }

      // Handle axios/fetch errors
      if (typeof err === "object" && err !== null) {
        const errorObj = err as any
        
        // Axios error format
        if (errorObj.response) {
          statusCode = errorObj.response.status
          const responseData = errorObj.response.data
          
          // Extract error message from response if available
          if (responseData && typeof responseData === "object") {
            errorMessage = responseData.message || responseData.error || errorMessage
          }
          
          // Rate limiting errors are retryable
          retryable = statusCode === 429
        } 
        // Fetch error format
        else if (errorObj.status) {
          statusCode = errorObj.status
          retryable = statusCode === 429
          
          // Try to parse the body for more details
          if (errorObj.json) {
            errorObj.json().then((data: any) => {
              if (data && data.message) {
                errorMessage = data.message
                setError({ 
                  message: errorMessage, 
                  statusCode, 
                  path: errorObj.url,
                  timestamp: new Date().toISOString(),
                  retryable 
                })
              }
            }).catch(() => {
              // Parsing failed, use default error
            })
          }
        }
      }

      const apiError: ApiError = {
        message: errorMessage,
        statusCode,
        path: typeof window !== 'undefined' ? window.location.pathname : undefined,
        timestamp: new Date().toISOString(),
        retryable
      }

      // Show toast notification for the error
      toast({
        title: statusCode ? `Error ${statusCode}` : "Error",
        description: errorMessage,
        variant: "destructive",
      })

      setError(apiError)
      return apiError
    },
    [toast]
  )

  // Wrapper for async API calls to handle loading state and errors
  const apiCallWrapper = useCallback(
    async <T>(
      apiCall: () => Promise<T>,
      options?: {
        loadingState?: boolean
        errorMessage?: string
        onSuccess?: (data: T) => void
      }
    ): Promise<T | null> => {
      const { loadingState = true, errorMessage, onSuccess } = options || {}

      if (loadingState) {
        setIsLoading(true)
      }

      try {
        const data = await apiCall()
        if (onSuccess) {
          onSuccess(data)
        }
        setError(null)
        return data
      } catch (err) {
        handleError(err, errorMessage)
        return null
      } finally {
        if (loadingState) {
          setIsLoading(false)
        }
      }
    },
    [handleError]
  )

  // Clear the current error
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    error,
    isLoading,
    handleError,
    apiCallWrapper,
    clearError,
  }
} 