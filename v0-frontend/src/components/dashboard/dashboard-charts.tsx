"use client"

import { useTheme } from "next-themes"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { useThreatTrends } from "@/hooks/api/use-threat-data"
import { Loader2 } from "lucide-react"

export function DashboardCharts() {
  const { theme } = useTheme()
  const isDark = theme === "dark" || true // Force dark theme to match the image

  const textColor = "#f8fafc" // Light text for dark theme
  const gridColor = "#334155" // Dark grid lines for dark theme

  const { trends: threatData, loading: trendsLoading, error: trendsError } = useThreatTrends(7)

  const modelAccuracyData = [
    { name: "Jan", accuracy: 78 },
    { name: "Feb", accuracy: 82 },
    { name: "Mar", accuracy: 85 },
    { name: "Apr", accuracy: 87 },
    { name: "May", accuracy: 89 },
    { name: "Jun", accuracy: 91 },
    { name: "Jul", accuracy: 93 },
  ]

  if (trendsLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        {Array(2)
          .fill(0)
          .map((_, i) => (
            <Card key={i} className="bg-[#0d1117] border-gray-800">
              <CardHeader>
                <CardTitle>Loading...</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-center py-16">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </CardContent>
            </Card>
          ))}
      </div>
    )
  }

  if (trendsError) {
    return (
      <div className="rounded-md bg-red-900/20 p-4 border border-red-800">
        <p className="text-red-400">{trendsError}</p>
      </div>
    )
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader>
          <CardTitle>Threat Detection</CardTitle>
          <CardDescription>Number of threats detected over time</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="line">
            <TabsList className="mb-4 bg-[#0f1117]">
              <TabsTrigger value="line" className="data-[state=active]:bg-[#2a2d3a]">
                Line
              </TabsTrigger>
              <TabsTrigger value="bar" className="data-[state=active]:bg-[#2a2d3a]">
                Bar
              </TabsTrigger>
            </TabsList>
            <TabsContent value="line">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={threatData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="date" stroke={textColor} />
                  <YAxis stroke={textColor} />
                  <Tooltip contentStyle={{ backgroundColor: "#1a1d29", borderColor: "#2a2d3a" }} />
                  <Legend />
                  <Line type="monotone" dataKey="count" stroke="#3b82f6" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </TabsContent>
            <TabsContent value="bar">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={threatData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="date" stroke={textColor} />
                  <YAxis stroke={textColor} />
                  <Tooltip contentStyle={{ backgroundColor: "#1a1d29", borderColor: "#2a2d3a" }} />
                  <Legend />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader>
          <CardTitle>AI Model Accuracy</CardTitle>
          <CardDescription>Model accuracy over time</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="line">
            <TabsList className="mb-4 bg-[#0f1117]">
              <TabsTrigger value="line" className="data-[state=active]:bg-[#2a2d3a]">
                Line
              </TabsTrigger>
              <TabsTrigger value="bar" className="data-[state=active]:bg-[#2a2d3a]">
                Bar
              </TabsTrigger>
            </TabsList>
            <TabsContent value="line">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={modelAccuracyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="name" stroke={textColor} />
                  <YAxis stroke={textColor} />
                  <Tooltip contentStyle={{ backgroundColor: "#1a1d29", borderColor: "#2a2d3a" }} />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#10b981" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </TabsContent>
            <TabsContent value="bar">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelAccuracyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="name" stroke={textColor} />
                  <YAxis stroke={textColor} />
                  <Tooltip contentStyle={{ backgroundColor: "#1a1d29", borderColor: "#2a2d3a" }} />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
