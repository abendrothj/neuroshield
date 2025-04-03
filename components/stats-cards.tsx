import type React from "react"
import { Card, CardContent } from "@/components/ui/card"
import { ShieldCheck, Brain, Activity } from "lucide-react"

export default function StatsCards() {
  return (
    <>
      <StatCard
        title="Threats Blocked Today"
        value="12"
        icon={<ShieldCheck className="h-5 w-5 text-primary" />}
        trend="+3 from yesterday"
        trendUp={true}
      />
      <StatCard
        title="Active Predictions"
        value="3"
        icon={<Brain className="h-5 w-5 text-secondary" />}
        trend="Monitoring 5 potential threats"
        trendUp={false}
      />
      <StatCard
        title="System Health"
        value="Stable"
        icon={<Activity className="h-5 w-5 text-green-500" />}
        statusDot={true}
        trend="All systems operational"
        trendUp={true}
      />
    </>
  )
}

function StatCard({
  title,
  value,
  icon,
  statusDot = false,
  trend,
  trendUp,
}: {
  title: string
  value: string
  icon: React.ReactNode
  statusDot?: boolean
  trend: string
  trendUp?: boolean
}) {
  return (
    <Card className="glass-effect hover:border-primary/50 transition-all duration-300">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="rounded-full p-2 bg-muted/50">{icon}</div>
          <div className="text-xs font-medium text-muted-foreground">{title}</div>
        </div>
        <div className="mt-4 flex items-center">
          <div className="text-3xl font-bold">{value}</div>
          {statusDot && <div className="ml-2 h-3 w-3 rounded-full bg-green-500 animate-pulse" />}
        </div>
        <div className="mt-2 text-xs text-muted-foreground flex items-center">
          {trendUp !== undefined && (
            <span className={`mr-1 ${trendUp ? "text-green-500" : "text-amber-500"}`}>{trendUp ? "↑" : "→"}</span>
          )}
          {trend}
        </div>
      </CardContent>
    </Card>
  )
}

