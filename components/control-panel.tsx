"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Zap, Shield, Database, Cpu, RefreshCw } from "lucide-react"
import { Progress } from "@/components/ui/progress"

interface ControlPanelProps {
  onAIResponse: () => void
}

export default function ControlPanel({ onAIResponse }: ControlPanelProps) {
  const [autoResponse, setAutoResponse] = useState(false)
  const [progress, setProgress] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)

  useEffect(() => {
    if (autoResponse) {
      const interval = setInterval(() => {
        onAIResponse()
      }, 30000) // Every 30 seconds

      return () => clearInterval(interval)
    }
  }, [autoResponse, onAIResponse])

  const handleTriggerResponse = () => {
    setIsProcessing(true)
    setProgress(0)

    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsProcessing(false)
          onAIResponse()
          return 0
        }
        return prev + 10
      })
    }, 100)
  }

  return (
    <div className="space-y-6">
      <Button
        className="w-full h-12 bg-primary hover:bg-primary/80 text-primary-foreground transition-all duration-300 relative overflow-hidden"
        onClick={handleTriggerResponse}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <>
            <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
            Processing...
            <Progress value={progress} className="absolute bottom-0 left-0 right-0 h-1 bg-transparent" />
          </>
        ) : (
          <>
            <Zap className="mr-2 h-5 w-5" />
            Trigger AI Response
          </>
        )}
      </Button>

      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <Label htmlFor="auto-response" className="text-sm">
            Auto-Response Mode
          </Label>
          <p className="text-xs text-muted-foreground">AI will respond automatically to threats</p>
        </div>
        <Switch
          id="auto-response"
          checked={autoResponse}
          onCheckedChange={setAutoResponse}
          className="data-[state=checked]:bg-primary"
        />
      </div>

      <div className="space-y-3 pt-2">
        <h3 className="text-sm font-medium">System Status</h3>
        <div className="grid grid-cols-2 gap-3">
          <StatusItem label="AI Engine" status="Online" color="green" icon={<Cpu className="h-4 w-4" />} />
          <StatusItem label="Blockchain" status="Synced" color="green" icon={<Database className="h-4 w-4" />} />
          <StatusItem label="Firewall" status="Active" color="green" icon={<Shield className="h-4 w-4" />} />
          <StatusItem label="Updates" status="Current" color="blue" icon={<RefreshCw className="h-4 w-4" />} />
        </div>
      </div>
    </div>
  )
}

function StatusItem({
  label,
  status,
  color,
  icon,
}: {
  label: string
  status: string
  color: "green" | "yellow" | "red" | "blue"
  icon: React.ReactNode
}) {
  const colorMap = {
    green: "bg-green-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
    blue: "bg-blue-500",
  }

  return (
    <div className="flex items-center justify-between p-3 rounded-md bg-muted/30 border border-white/5">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className="flex items-center">
        <span className="text-xs mr-1.5">{status}</span>
        <div className={`h-2 w-2 rounded-full ${colorMap[color]}`} />
      </div>
    </div>
  )
}

