import { Suspense } from "react"
import { AIModelsHeader } from "@/components/ai-models/ai-models-header"
import { AIModelsList } from "@/components/ai-models/ai-models-list"
import { AIModelsStats } from "@/components/ai-models/ai-models-stats"
import { AIModelsSkeleton } from "@/components/ai-models/ai-models-skeleton"

export default function AIModelsPage() {
  return (
    <div className="space-y-6 ml-64">
      <AIModelsHeader />
      <Suspense fallback={<AIModelsSkeleton />}>
        <AIModelsStats />
        <AIModelsList />
      </Suspense>
    </div>
  )
}
