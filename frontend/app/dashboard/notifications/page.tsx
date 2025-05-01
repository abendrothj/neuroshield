import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Shield, Database, Brain, HardDrive, Settings, Info, AlertTriangle } from "lucide-react"

export default function NotificationsPage() {
  return (
    <div className="space-y-6 ml-64">
      <div>
        <h1 className="text-3xl font-bold">Notifications</h1>
        <p className="text-muted-foreground">View and manage your notifications</p>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList className="mb-4 bg-[#0f1117]">
          <TabsTrigger value="all" className="data-[state=active]:bg-[#2a2d3a]">
            All
          </TabsTrigger>
          <TabsTrigger value="alerts" className="data-[state=active]:bg-[#2a2d3a]">
            Alerts
          </TabsTrigger>
          <TabsTrigger value="system" className="data-[state=active]:bg-[#2a2d3a]">
            System
          </TabsTrigger>
          <TabsTrigger value="updates" className="data-[state=active]:bg-[#2a2d3a]">
            Updates
          </TabsTrigger>
        </TabsList>

        <TabsContent value="all">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>All Notifications</CardTitle>
              <CardDescription>View all your recent notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    icon: Shield,
                    title: "Critical Threat Detected",
                    description: "A critical threat was detected in your network.",
                    time: "Just now",
                    type: "alert",
                  },
                  {
                    icon: Database,
                    title: "Blockchain Transaction Confirmed",
                    description: "Your recent blockchain transaction has been confirmed.",
                    time: "2 hours ago",
                    type: "system",
                  },
                  {
                    icon: Brain,
                    title: "AI Model Training Complete",
                    description: "The ThreatDetector model has completed training with 96.5% accuracy.",
                    time: "Yesterday",
                    type: "update",
                  },
                  {
                    icon: HardDrive,
                    title: "IPFS Storage Limit Warning",
                    description: "You are approaching your IPFS storage limit (85% used).",
                    time: "2 days ago",
                    type: "alert",
                  },
                  {
                    icon: Settings,
                    title: "System Maintenance Scheduled",
                    description: "System maintenance is scheduled for April 30, 2023 at 2:00 AM UTC.",
                    time: "3 days ago",
                    type: "system",
                  },
                ].map((notification, index) => (
                  <div
                    key={index}
                    className="flex items-start space-x-4 border-b border-gray-800 pb-4 last:border-0 last:pb-0"
                  >
                    <div
                      className={`rounded-full p-2 ${
                        notification.type === "alert"
                          ? "bg-red-900/20 text-red-400"
                          : notification.type === "system"
                            ? "bg-blue-900/20 text-blue-400"
                            : "bg-green-900/20 text-green-400"
                      }`}
                    >
                      <notification.icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-medium">{notification.title}</h4>
                      <p className="text-sm text-muted-foreground">{notification.description}</p>
                      <p className="text-xs text-muted-foreground mt-1">{notification.time}</p>
                    </div>
                    <Button variant="ghost" size="sm">
                      Mark as Read
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>Alert Notifications</CardTitle>
              <CardDescription>Security and system alerts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    icon: Shield,
                    title: "Critical Threat Detected",
                    description: "A critical threat was detected in your network.",
                    time: "Just now",
                  },
                  {
                    icon: HardDrive,
                    title: "IPFS Storage Limit Warning",
                    description: "You are approaching your IPFS storage limit (85% used).",
                    time: "2 days ago",
                  },
                  {
                    icon: AlertTriangle,
                    title: "Unusual Login Attempt",
                    description: "Unusual login attempt detected from IP 203.45.67.89.",
                    time: "1 week ago",
                  },
                ].map((notification, index) => (
                  <div
                    key={index}
                    className="flex items-start space-x-4 border-b border-gray-800 pb-4 last:border-0 last:pb-0"
                  >
                    <div className="bg-red-900/20 text-red-400 rounded-full p-2">
                      <notification.icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-medium">{notification.title}</h4>
                      <p className="text-sm text-muted-foreground">{notification.description}</p>
                      <p className="text-xs text-muted-foreground mt-1">{notification.time}</p>
                    </div>
                    <Button variant="ghost" size="sm">
                      Mark as Read
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>System Notifications</CardTitle>
              <CardDescription>System status and updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    icon: Database,
                    title: "Blockchain Transaction Confirmed",
                    description: "Your recent blockchain transaction has been confirmed.",
                    time: "2 hours ago",
                  },
                  {
                    icon: Settings,
                    title: "System Maintenance Scheduled",
                    description: "System maintenance is scheduled for April 30, 2023 at 2:00 AM UTC.",
                    time: "3 days ago",
                  },
                  {
                    icon: Info,
                    title: "System Update Completed",
                    description: "System update to version 2.4.0 completed successfully.",
                    time: "1 week ago",
                  },
                ].map((notification, index) => (
                  <div
                    key={index}
                    className="flex items-start space-x-4 border-b border-gray-800 pb-4 last:border-0 last:pb-0"
                  >
                    <div className="bg-blue-900/20 text-blue-400 rounded-full p-2">
                      <notification.icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-medium">{notification.title}</h4>
                      <p className="text-sm text-muted-foreground">{notification.description}</p>
                      <p className="text-xs text-muted-foreground mt-1">{notification.time}</p>
                    </div>
                    <Button variant="ghost" size="sm">
                      Mark as Read
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="updates">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>Update Notifications</CardTitle>
              <CardDescription>Feature and model updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    icon: Brain,
                    title: "AI Model Training Complete",
                    description: "The ThreatDetector model has completed training with 96.5% accuracy.",
                    time: "Yesterday",
                  },
                  {
                    icon: Shield,
                    title: "New Threat Signatures Added",
                    description: "250 new threat signatures have been added to the database.",
                    time: "5 days ago",
                  },
                  {
                    icon: Info,
                    title: "New Feature: Advanced Analytics",
                    description: "Advanced analytics dashboard is now available in your account.",
                    time: "2 weeks ago",
                  },
                ].map((notification, index) => (
                  <div
                    key={index}
                    className="flex items-start space-x-4 border-b border-gray-800 pb-4 last:border-0 last:pb-0"
                  >
                    <div className="bg-green-900/20 text-green-400 rounded-full p-2">
                      <notification.icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-medium">{notification.title}</h4>
                      <p className="text-sm text-muted-foreground">{notification.description}</p>
                      <p className="text-xs text-muted-foreground mt-1">{notification.time}</p>
                    </div>
                    <Button variant="ghost" size="sm">
                      Mark as Read
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
