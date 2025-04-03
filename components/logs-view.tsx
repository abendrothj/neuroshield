"use client"

import { useState, useEffect } from "react"
import BlockchainLogs from "./blockchain-logs"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Download, Filter, RefreshCw, Calendar } from "lucide-react"
import { Input } from "@/components/ui/input"

export default function LogsView() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  const handleRefresh = () => {
    setIsRefreshing(true)
    // Simulate refresh
    setTimeout(() => {
      setIsRefreshing(false)
    }, 1000)
  }

  return (
    <div className={`space-y-6 transition-opacity duration-500 ${isLoaded ? "opacity-100" : "opacity-0"}`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <h1 className="text-2xl font-bold">Security Logs</h1>
        <div className="flex flex-wrap gap-2">
          <div className="relative">
            <Calendar className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input type="date" className="pl-8 bg-muted/50 border-white/10 focus-visible:ring-primary h-9" />
          </div>
          <Button
            variant="outline"
            size="sm"
            className="border-white/10 text-muted-foreground hover:text-primary hover:border-primary"
          >
            <Filter className="mr-2 h-4 w-4" />
            Filter
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="border-white/10 text-muted-foreground hover:text-primary hover:border-primary"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="border-white/10 text-muted-foreground hover:text-primary hover:border-primary"
          >
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      <Tabs defaultValue="blockchain" className="w-full">
        <TabsList className="w-full bg-muted/50 border border-white/10">
          <TabsTrigger
            value="blockchain"
            className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            Blockchain Logs
          </TabsTrigger>
          <TabsTrigger
            value="system"
            className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            System Logs
          </TabsTrigger>
          <TabsTrigger
            value="auth"
            className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            Authentication Logs
          </TabsTrigger>
        </TabsList>
        <TabsContent value="blockchain" className="mt-4">
          <BlockchainLogs />
        </TabsContent>
        <TabsContent value="system" className="mt-4">
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="text-xl">System Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-4 text-muted-foreground text-center">
                <p>System logs will appear here.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="auth" className="mt-4">
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="text-xl">Authentication Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-4 text-muted-foreground text-center">
                <p>Authentication logs will appear here.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

