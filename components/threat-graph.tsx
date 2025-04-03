"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Info } from "lucide-react"
import { CartesianGrid, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, Area, AreaChart } from "recharts"

// Generate 24 hours of dummy threat data
const generateThreatData = () => {
  const data = []
  const now = new Date()

  for (let i = 23; i >= 0; i--) {
    const time = new Date(now)
    time.setHours(now.getHours() - i)

    data.push({
      time: time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      traffic: Math.floor(Math.random() * 15) + 5,
      threats: Math.floor(Math.random() * 10) + 1,
      blocked: Math.floor(Math.random() * 8),
    })
  }

  return data
}

export default function ThreatGraph() {
  const [data] = useState(generateThreatData())

  return (
    <Card className="glass-effect">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">Threat Activity</CardTitle>
          <div className="relative group">
            <Info className="h-5 w-5 text-muted-foreground cursor-help" />
            <div className="absolute right-0 w-64 p-2 mt-2 text-xs glass-effect rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-10 border border-white/10">
              Real-time threat activity detected over the last 24 hours. Spikes indicate potential coordinated attacks.
            </div>
          </div>
        </div>
        <CardDescription>Last 24 hours of network activity</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <defs>
                <linearGradient id="colorTraffic" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(30, 100%, 50%)" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="hsl(30, 100%, 50%)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorThreats" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(196, 100%, 47%)" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="hsl(196, 100%, 47%)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorBlocked" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(270, 100%, 60%)" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="hsl(270, 100%, 60%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="time" stroke="#666" tick={{ fill: "#999" }} />
              <YAxis stroke="#666" tick={{ fill: "#999" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(0, 0, 0, 0.8)",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  borderRadius: "4px",
                  boxShadow: "0 4px 6px rgba(0, 0, 0, 0.3)",
                }}
                formatter={(value, name) => {
                  const formattedName =
                    name === "traffic"
                      ? "Incoming Traffic"
                      : name === "threats"
                        ? "Detected Threats"
                        : "Blocked Threats"
                  return [value, formattedName]
                }}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <Legend
                verticalAlign="top"
                height={36}
                formatter={(value) => {
                  return value === "traffic"
                    ? "Incoming Traffic"
                    : value === "threats"
                      ? "Detected Threats"
                      : "Blocked Threats"
                }}
              />
              <Area
                type="monotone"
                dataKey="traffic"
                stroke="hsl(30, 100%, 50%)"
                fillOpacity={1}
                fill="url(#colorTraffic)"
                strokeWidth={2}
                name="traffic"
                activeDot={{ r: 5, stroke: "#fff", strokeWidth: 1 }}
              />
              <Area
                type="monotone"
                dataKey="threats"
                stroke="hsl(196, 100%, 47%)"
                fillOpacity={1}
                fill="url(#colorThreats)"
                strokeWidth={2}
                name="threats"
                activeDot={{ r: 5, stroke: "#fff", strokeWidth: 1 }}
              />
              <Area
                type="monotone"
                dataKey="blocked"
                stroke="hsl(270, 100%, 60%)"
                fillOpacity={1}
                fill="url(#colorBlocked)"
                strokeWidth={2}
                name="blocked"
                activeDot={{ r: 5, stroke: "#fff", strokeWidth: 1 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

