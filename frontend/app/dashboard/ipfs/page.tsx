import { Suspense } from "react"
import { IPFSHeader } from "@/components/ipfs/ipfs-header"
import { IPFSFiles } from "@/components/ipfs/ipfs-files"
import { IPFSStats } from "@/components/ipfs/ipfs-stats"
import { IPFSSkeleton } from "@/components/ipfs/ipfs-skeleton"

export default function IPFSPage() {
  return (
    <div className="space-y-6 ml-64">
      <IPFSHeader />
      <Suspense fallback={<IPFSSkeleton />}>
        <IPFSStats />
        <IPFSFiles />
      </Suspense>
    </div>
  )
}
