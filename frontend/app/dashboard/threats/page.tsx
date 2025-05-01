import { Suspense } from "react"
import { ThreatsList } from "@/components/threats/threats-list"
import { ThreatsHeader } from "@/components/threats/threats-header"
import { ThreatsFilters } from "@/components/threats/threats-filters"
import { ThreatsListSkeleton } from "@/components/threats/threats-list-skeleton"

export default function ThreatsPage() {
  return (
    <div className="space-y-6 ml-64">
      <ThreatsHeader />
      <ThreatsFilters />
      <Suspense fallback={<ThreatsListSkeleton />}>
        <ThreatsList />
      </Suspense>
    </div>
  )
}
