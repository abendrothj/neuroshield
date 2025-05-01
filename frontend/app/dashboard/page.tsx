import { Suspense } from "react"
import { DashboardOverview } from "@/components/dashboard/dashboard-overview"
import { DashboardCards } from "@/components/dashboard/dashboard-cards"
import { DashboardCharts } from "@/components/dashboard/dashboard-charts"
import { DashboardSkeleton } from "@/components/dashboard/dashboard-skeleton"

export default function DashboardPage() {
  return (
    <div className="space-y-6 ml-64">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardOverview />
        <DashboardCards />
        <DashboardCharts />
      </Suspense>
    </div>
  )
}
