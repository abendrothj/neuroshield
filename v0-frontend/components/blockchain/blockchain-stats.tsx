import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getBlockchainStatus } from "@/lib/api/system"
import { Database, Network, Layers, Activity } from "lucide-react"

export async function BlockchainStats() {
  const blockchain = await getBlockchainStatus()

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Network</CardTitle>
          <Network className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-xl font-bold">Hyperledger Fabric</div>
          <p className="text-xs text-muted-foreground">Active network</p>
        </CardContent>
      </Card>
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Peers</CardTitle>
          <Database className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-xl font-bold">5</div>
          <p className="text-xs text-muted-foreground">Connected peers</p>
        </CardContent>
      </Card>
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Blocks</CardTitle>
          <Layers className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-xl font-bold">12458</div>
          <p className="text-xs text-muted-foreground">Total blocks</p>
        </CardContent>
      </Card>
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Status</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-xl font-bold text-green-500">Healthy</div>
          <p className="text-xs text-muted-foreground">Last updated: 2 minutes ago</p>
        </CardContent>
      </Card>
    </div>
  )
}
