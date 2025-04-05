import React from 'react'
import { AlertCircle, CheckCircle, Info, X, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface AlertProps {
  type: 'error' | 'success' | 'warning' | 'info'
  title?: string
  message: string
  onClose?: () => void
}

const Alert: React.FC<AlertProps> = ({ 
  type, 
  title, 
  message, 
  onClose 
}) => {
  const defaultTitles = {
    error: "Error",
    success: "Success",
    warning: "Warning",
    info: "Information"
  }

  const icons = {
    error: <AlertCircle className="h-4 w-4" />,
    success: <CheckCircle className="h-4 w-4" />,
    warning: <AlertTriangle className="h-4 w-4" />,
    info: <Info className="h-4 w-4" />
  }

  const colorClasses = {
    error: "text-red-700 bg-red-50 border-red-200 dark:text-red-300 dark:bg-red-900/20 dark:border-red-800",
    success: "text-green-700 bg-green-50 border-green-200 dark:text-green-300 dark:bg-green-900/20 dark:border-green-800",
    warning: "text-amber-700 bg-amber-50 border-amber-200 dark:text-amber-300 dark:bg-amber-900/20 dark:border-amber-800",
    info: "text-blue-700 bg-blue-50 border-blue-200 dark:text-blue-300 dark:bg-blue-900/20 dark:border-blue-800"
  }

  return (
    <div className={`p-4 rounded-lg border ${colorClasses[type]}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start">
          <span className="mr-2 mt-0.5">{icons[type]}</span>
          <div>
            <h5 className="font-medium">{title || defaultTitles[type]}</h5>
            <div className="text-sm">{message}</div>
          </div>
        </div>
        {onClose && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0 rounded-full -mt-1 -mr-1"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Close</span>
          </Button>
        )}
      </div>
    </div>
  )
}

export default Alert
