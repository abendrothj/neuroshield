import { Suspense } from "react"
import { BlockchainEvents } from "@/components/blockchain/blockchain-events"
import { Skeleton } from "@/components/ui/skeleton"

export default function BlockchainDashboard() {
  return (
    <div className="space-y-6 ml-64">
      <h1 className="text-3xl font-bold">Blockchain Security Events</h1>
      <p className="text-muted-foreground">
        View and manage immutable security event records stored on the blockchain
      </p>
      
      <Suspense fallback={<Skeleton className="h-[600px] w-full" />}>
        <BlockchainEvents />
      </Suspense>
    </div>
  )
}
