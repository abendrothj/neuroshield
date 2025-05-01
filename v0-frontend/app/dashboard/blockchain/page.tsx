import { Suspense } from "react"
import { BlockchainTransactions } from "@/components/blockchain/blockchain-transactions"
import { BlockchainHeader } from "@/components/blockchain/blockchain-header"
import { BlockchainStats } from "@/components/blockchain/blockchain-stats"
import { BlockchainSkeleton } from "@/components/blockchain/blockchain-skeleton"

export default function BlockchainPage() {
  return (
    <div className="space-y-6 ml-64">
      <BlockchainHeader />
      <Suspense fallback={<BlockchainSkeleton />}>
        <BlockchainStats />
        <BlockchainTransactions />
      </Suspense>
    </div>
  )
}
