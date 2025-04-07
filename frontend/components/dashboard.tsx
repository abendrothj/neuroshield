"use client"

import { useState, useEffect } from "react"
import ThreatGraph from "./threat-graph"
import StatsCards from "./stats-cards"
import BlockchainLogs from "./blockchain-logs"
import ControlPanel from "./control-panel"
import SecurityEvents from "./SecurityEvents"
import SocketClient from "@/lib/socket"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Shield, Zap, Activity, AlertTriangle } from "lucide-react"

export default function Dashboard() {
  const [lastAction, setLastAction] = useState("None")
  const [isLoaded, setIsLoaded] = useState(false)
  const socket = SocketClient.getInstance()

  useEffect(() => {
    // Subscribe to various events
    socket.subscribe("threatDetected", (data) => {
      console.log("Threat detected:", data)
      setLastAction(`Threat detected: ${data.type} at ${new Date().toLocaleString()}`)
    })

    socket.subscribe("systemMetrics", (data) => {
      console.log("System metrics updated:", data)
    })

    socket.subscribe("blockchainUpdate", (data) => {
      console.log("Blockchain updated:", data)
    })

    setIsLoaded(true)

    // Cleanup on unmount
    return () => {
      socket.unsubscribe("threatDetected", () => {})
      socket.unsubscribe("systemMetrics", () => {})
      socket.unsubscribe("blockchainUpdate", () => {})
    }
  }, [])

  const handleAIResponse = () => {
    socket.emit("triggerAIResponse", { timestamp: new Date().toISOString() })
    setLastAction(`AI Response Triggered at ${new Date().toLocaleString()}`)
  }

  return (
    <div className={`space-y-6 transition-opacity duration-500 ${isLoaded ? "opacity-100" : "opacity-0"}`}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCards />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Tabs defaultValue="threats" className="w-full">
            <TabsList className="w-full bg-muted/50 border border-white/10">
              <TabsTrigger
                value="threats"
                className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
              >
                <Shield className="mr-2 h-4 w-4" />
                Threat Activity
              </TabsTrigger>
              <TabsTrigger
                value="system"
                className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
              >
                <Activity className="mr-2 h-4 w-4" />
                System Health
              </TabsTrigger>
              <TabsTrigger
                value="alerts"
                className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
              >
                <AlertTriangle className="mr-2 h-4 w-4" />
                Alerts
              </TabsTrigger>
            </TabsList>
            <TabsContent value="threats" className="mt-4">
              <ThreatGraph />
            </TabsContent>
            <TabsContent value="system" className="mt-4">
              <Card className="glass-effect">
                <CardHeader>
                  <CardTitle>System Health</CardTitle>
                  <CardDescription>Real-time system performance metrics</CardDescription>
                </CardHeader>
                <CardContent className="h-[300px] flex items-center justify-center">
                  <p className="text-muted-foreground">System health metrics will appear here</p>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="alerts" className="mt-4">
              <SecurityEvents />
            </TabsContent>
          </Tabs>
        </div>

        <div className="glass-effect rounded-lg p-6 border border-white/10">
          <div className="flex items-center mb-4">
            <Zap className="h-5 w-5 text-primary mr-2" />
            <h3 className="text-lg font-medium">AI Control Center</h3>
          </div>
          <ControlPanel onAIResponse={handleAIResponse} />
        </div>
      </div>

      <BlockchainLogs />

      <div className="text-sm text-muted-foreground">
        <span className="font-medium">Last Action:</span> {lastAction}
      </div>
    </div>
  )
}

