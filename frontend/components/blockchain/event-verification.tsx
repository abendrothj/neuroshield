'use client'

import { useState } from 'react'
import { 
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle 
} from "../ui/card"
import { 
  Alert,
  AlertTitle,
  AlertDescription
} from "../ui/alert"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Label } from "../ui/label"
import { 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  Shield, 
  Clock,
  FileText,
  Hash 
} from "lucide-react"
import { Skeleton } from "../ui/skeleton"

type VerificationStatus = 'verified' | 'invalid' | 'pending' | 'error'

interface VerificationCertificate {
  eventId: string
  type: string
  timestamp: string
  blockchainTimestamp: string
  blockNumber: string | number
  transactionId: string
  dataHash: string
  verificationTimestamp: string
  status: string
  verificationDetails: {
    dataIntegrity: {
      isValid: boolean
      calculatedHash: string
      storedHash: string
    }
    transactionValidity: {
      isValid: boolean
      txId: string
      blockNumber: string | number
      timestamp: string
      validationCode: number
    }
  }
}

export default function EventVerification() {
  const [eventId, setEventId] = useState('')
  const [status, setStatus] = useState<VerificationStatus>('pending')
  const [isLoading, setIsLoading] = useState(false)
  const [certificate, setCertificate] = useState<VerificationCertificate | null>(null)
  const [error, setError] = useState('')
  
  const verifyEvent = async () => {
    if (!eventId) return
    
    setIsLoading(true)
    setStatus('pending')
    setError('')
    
    try {
      // Make API call to backend verification endpoint
      const response = await fetch(`/api/blockchain/verify/${eventId}`)
      
      if (!response.ok) {
        throw new Error(`Verification failed: ${response.statusText}`)
      }
      
      const data = await response.json()
      setCertificate(data)
      
      // Set status based on verification result
      if (data.status === 'VERIFIED' && 
          data.verificationDetails.dataIntegrity.isValid && 
          data.verificationDetails.transactionValidity.isValid) {
        setStatus('verified')
      } else {
        setStatus('invalid')
      }
    } catch (err) {
      console.error('Verification error:', err)
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
      setStatus('error')
    } finally {
      setIsLoading(false)
    }
  }
  
  const renderStatusIcon = () => {
    switch (status) {
      case 'verified':
        return <CheckCircle2 className="h-16 w-16 text-green-500" />
      case 'invalid':
        return <XCircle className="h-16 w-16 text-red-500" />
      case 'error':
        return <AlertTriangle className="h-16 w-16 text-orange-500" />
      case 'pending':
      default:
        return <Shield className="h-16 w-16 text-gray-400" />
    }
  }
  
  const renderStatusMessage = () => {
    switch (status) {
      case 'verified':
        return (
          <Alert className="bg-green-50 border-green-200">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-800">Verification Successful</AlertTitle>
            <AlertDescription className="text-green-700">
              This event has been verified as authentic and unmodified.
            </AlertDescription>
          </Alert>
        )
      case 'invalid':
        return (
          <Alert className="bg-red-50 border-red-200">
            <XCircle className="h-4 w-4 text-red-600" />
            <AlertTitle className="text-red-800">Verification Failed</AlertTitle>
            <AlertDescription className="text-red-700">
              This event could not be verified. The data may have been tampered with.
            </AlertDescription>
          </Alert>
        )
      case 'error':
        return (
          <Alert className="bg-orange-50 border-orange-200">
            <AlertTriangle className="h-4 w-4 text-orange-600" />
            <AlertTitle className="text-orange-800">Verification Error</AlertTitle>
            <AlertDescription className="text-orange-700">
              {error || 'An error occurred during verification.'}
            </AlertDescription>
          </Alert>
        )
      case 'pending':
      default:
        return null
    }
  }
  
  const renderCertificate = () => {
    if (!certificate) return null
    
    return (
      <div className="mt-6 space-y-4">
        <h3 className="text-lg font-medium">Verification Certificate</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center">
              <FileText className="h-4 w-4 mr-2 text-blue-500" />
              <span className="text-sm font-medium">Event Type:</span>
              <span className="ml-2 text-sm">{certificate.type}</span>
            </div>
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-2 text-blue-500" />
              <span className="text-sm font-medium">Event Time:</span>
              <span className="ml-2 text-sm">
                {new Date(certificate.timestamp).toLocaleString()}
              </span>
            </div>
            <div className="flex items-center">
              <Hash className="h-4 w-4 mr-2 text-blue-500" />
              <span className="text-sm font-medium">Data Hash:</span>
              <span className="ml-2 text-sm text-gray-500">
                {certificate.dataHash.substring(0, 8)}...{certificate.dataHash.substring(certificate.dataHash.length - 8)}
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center">
              <span className="text-sm font-medium">Block Number:</span>
              <span className="ml-2 text-sm">{certificate.blockNumber}</span>
            </div>
            <div className="flex items-center">
              <span className="text-sm font-medium">Transaction ID:</span>
              <span className="ml-2 text-sm text-gray-500">
                {certificate.transactionId.substring(0, 8)}...{certificate.transactionId.substring(certificate.transactionId.length - 8)}
              </span>
            </div>
            <div className="flex items-center">
              <span className="text-sm font-medium">Blockchain Time:</span>
              <span className="ml-2 text-sm">
                {new Date(certificate.blockchainTimestamp).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center">
          <Shield className="h-5 w-5 mr-2" />
          Event Verification
        </CardTitle>
        <CardDescription>
          Verify the authenticity and integrity of security events
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="flex flex-col space-y-2">
            <Label htmlFor="event-id">Event ID</Label>
            <div className="flex space-x-2">
              <Input
                id="event-id"
                placeholder="Enter event ID to verify"
                value={eventId}
                onChange={(e) => setEventId(e.target.value)}
              />
              <Button 
                onClick={verifyEvent} 
                disabled={!eventId || isLoading}
              >
                Verify
              </Button>
            </div>
          </div>
          
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : (
            <>
              <div className="flex justify-center">
                {renderStatusIcon()}
              </div>
              
              {renderStatusMessage()}
              
              {status === 'verified' && renderCertificate()}
            </>
          )}
        </div>
      </CardContent>
      <CardFooter className="text-xs text-gray-500">
        Powered by NeuraShield blockchain verification
      </CardFooter>
    </Card>
  )
} 