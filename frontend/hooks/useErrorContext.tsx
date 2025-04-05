"use client"

import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react'
import { ApiError, useApiError } from './useApiError'
import Alert from '@/components/ui/alert'

interface ErrorContextProps {
  globalError: ApiError | null
  setGlobalError: (error: ApiError | null) => void
  clearGlobalError: () => void
  handleGlobalError: (error: unknown, customMessage?: string) => void
}

const ErrorContext = createContext<ErrorContextProps | undefined>(undefined)

export const ErrorProvider: React.FC<{children: ReactNode}> = ({ children }) => {
  const [globalError, setGlobalError] = useState<ApiError | null>(null)
  const { handleError } = useApiError()

  const clearGlobalError = useCallback(() => {
    setGlobalError(null)
  }, [])

  const handleGlobalError = useCallback(
    (error: unknown, customMessage?: string) => {
      const apiError = handleError(error, customMessage)
      setGlobalError(apiError)
      return apiError
    },
    [handleError]
  )

  return (
    <ErrorContext.Provider
      value={{
        globalError,
        setGlobalError,
        clearGlobalError,
        handleGlobalError,
      }}
    >
      {globalError && (
        <div className="fixed bottom-4 right-4 z-50 max-w-md">
          <Alert
            type="error"
            title={globalError.statusCode ? `Error ${globalError.statusCode}` : "Error"}
            message={globalError.message}
            onClose={clearGlobalError}
          />
        </div>
      )}
      {children}
    </ErrorContext.Provider>
  )
}

export const useErrorContext = () => {
  const context = useContext(ErrorContext)
  if (context === undefined) {
    throw new Error('useErrorContext must be used within an ErrorProvider')
  }
  return context
} 