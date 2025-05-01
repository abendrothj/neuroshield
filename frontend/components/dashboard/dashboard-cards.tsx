import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { getSystemHealth, getBlockchainStatus, getIPFSStatus } from "@/lib/api/system"

export async function DashboardCards() {
  const health = await getSystemHealth()
  const blockchain = await getBlockchainStatus()
  const ipfs = await getIPFSStatus()

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>Overall system performance and health</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">CPU Usage</div>
              <div className="text-sm">{health.cpu}%</div>
            </div>
            <div className="h-2 w-full rounded-full bg-gray-700">
              <div className="h-2 rounded-full bg-blue-500" style={{ width: `${health.cpu}%` }} />
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Memory Usage</div>
              <div className="text-sm">{health.memory}%</div>
            </div>
            <div className="h-2 w-full rounded-full bg-gray-700">
              <div className="h-2 rounded-full bg-blue-500" style={{ width: `${health.memory}%` }} />
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Disk Usage</div>
              <div className="text-sm">{health.disk}%</div>
            </div>
            <div className="h-2 w-full rounded-full bg-gray-700">
              <div className="h-2 rounded-full bg-blue-500" style={{ width: `${health.disk}%` }} />
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader>
          <CardTitle>Blockchain Status</CardTitle>
          <CardDescription>Current blockchain network status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Network</div>
              <div className="text-sm">{blockchain.network}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Peers</div>
              <div className="text-sm">{blockchain.peers}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Blocks</div>
              <div className="text-sm">{blockchain.blocks}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Last Block</div>
              <div className="text-sm">{blockchain.lastBlock}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Status</div>
              <div className={`text-sm ${blockchain.status === "Healthy" ? "text-green-500" : "text-red-500"}`}>
                {blockchain.status}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="bg-[#0d1117] border-gray-800">
        <CardHeader>
          <CardTitle>IPFS Storage</CardTitle>
          <CardDescription>IPFS network and storage statistics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Status</div>
              <div className={`text-sm ${ipfs.status === "Online" ? "text-green-500" : "text-red-500"}`}>
                {ipfs.status}
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Peers</div>
              <div className="text-sm">{ipfs.peers}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Files</div>
              <div className="text-sm">{ipfs.files}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Storage Used</div>
              <div className="text-sm">{ipfs.storageUsed}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Bandwidth</div>
              <div className="text-sm">{ipfs.bandwidth}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
