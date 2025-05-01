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

const threatData = [
  { name: "Jan", threats: 65 },
  { name: "Feb", threats: 59 },
  { name: "Mar", threats: 80 },
  { name: "Apr", threats: 81 },
  { name: "May", threats: 56 },
  { name: "Jun", threats: 55 },
  { name: "Jul", threats: 40 },
]

const modelAccuracyData = [
  { name: "Jan", accuracy: 78 },
  { name: "Feb", accuracy: 82 },
  { name: "Mar", accuracy: 85 },
  { name: "Apr", accuracy: 87 },
  { name: "May", accuracy: 89 },
  { name: "Jun", accuracy: 91 },
  { name: "Jul", accuracy: 93 },
]

export function DashboardCharts() {
  const { theme } = useTheme()
  const isDark = theme === "dark" || true // Force dark theme to match the image

  const textColor = "#f8fafc" // Light text for dark theme
  const gridColor = "#334155" // Dark grid lines for dark theme

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
                  <XAxis dataKey="name" stroke={textColor} />
                  <YAxis stroke={textColor} />
                  <Tooltip contentStyle={{ backgroundColor: "#1a1d29", borderColor: "#2a2d3a" }} />
                  <Legend />
                  <Line type="monotone" dataKey="threats" stroke="#3b82f6" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </TabsContent>
            <TabsContent value="bar">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={threatData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="name" stroke={textColor} />
                  <YAxis stroke={textColor} />
                  <Tooltip contentStyle={{ backgroundColor: "#1a1d29", borderColor: "#2a2d3a" }} />
                  <Legend />
                  <Bar dataKey="threats" fill="#3b82f6" />
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
