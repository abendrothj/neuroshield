"use client"

import { useState, useEffect } from "react"
import ThreatGraph from "./threat-graph"
import StatsCards from "./stats-cards"
import BlockchainLogs from "./blockchain-logs"
import ControlPanel from "./control-panel"
import SecurityEvents from "./SecurityEvents"
import SocketClient from "@/lib/socket"
import { api } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Shield, Zap, Activity, AlertTriangle } from "lucide-react"

export default function Dashboard() {
  const [lastAction, setLastAction] = useState("None")
  const [isLoaded, setIsLoaded] = useState(false)
  const [aiServiceHealth, setAIServiceHealth] = useState<any>(null)
  const socket = SocketClient.getInstance()

  useEffect(() => {
    // Check AI service health
    const checkAIHealth = async () => {
      try {
        const health = await api.getAIServiceHealth()
        setAIServiceHealth(health)
      } catch (error) {
        console.error("Failed to get AI service health:", error)
      }
    }

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

    checkAIHealth()
    setIsLoaded(true)

    // Cleanup on unmount
    return () => {
      socket.unsubscribe("threatDetected", () => {})
      socket.unsubscribe("systemMetrics", () => {})
      socket.unsubscribe("blockchainUpdate", () => {})
    }
  }, [])

  const handleAIResponse = async () => {
    try {
      // Example threat data - replace with real data from your system
      const threatData = {
        data: [{
          feature_1: 0.1, feature_2: 0.2, feature_3: 0.3, feature_4: 0.4, feature_5: 0.5,
          feature_6: 0.6, feature_7: 0.7, feature_8: 0.8, feature_9: 0.9, feature_10: 1.0,
          feature_11: 0.1, feature_12: 0.2, feature_13: 0.3, feature_14: 0.4, feature_15: 0.5,
          feature_16: 0.6, feature_17: 0.7, feature_18: 0.8, feature_19: 0.9, feature_20: 1.0,
          feature_21: 0.1, feature_22: 0.2, feature_23: 0.3, feature_24: 0.4, feature_25: 0.5,
          feature_26: 0.6, feature_27: 0.7, feature_28: 0.8, feature_29: 0.9, feature_30: 1.0,
          feature_31: 0.1, feature_32: 0.2, feature_33: 0.3, feature_34: 0.4, feature_35: 0.5,
          feature_36: 0.6, feature_37: 0.7, feature_38: 0.8, feature_39: 0.9
        }]
      }

      const prediction = await api.analyzeThreat(threatData)
      setLastAction(`AI Analysis: ${prediction[0].threat_level} (${(prediction[0].confidence * 100).toFixed(2)}% confidence) at ${new Date().toLocaleString()}`)
    } catch (error: any) {
      console.error("Failed to analyze threat:", error)
      setLastAction(`AI Analysis failed: ${error.message}`)
    }
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
                <CardContent className="h-[300px] flex flex-col space-y-4">
                  {aiServiceHealth && (
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">AI Service Status:</span>
                        <span className={aiServiceHealth.status === "healthy" ? "text-green-500" : "text-red-500"}>
                          {aiServiceHealth.status}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model Version:</span>
                        <span>{aiServiceHealth.model_version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model Loaded:</span>
                        <span>{aiServiceHealth.model_loaded ? "Yes" : "No"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">GPU Available:</span>
                        <span>{aiServiceHealth.gpu_available ? "Yes" : "No"}</span>
                      </div>
                    </div>
                  )}
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

