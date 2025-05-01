import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getBlockchainTransactions } from "@/lib/api/blockchain"

export async function BlockchainTransactions() {
  const transactions = await getBlockchainTransactions()

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Recent Transactions</h2>
        <Button variant="outline" size="sm">
          View All
        </Button>
      </div>
      <div className="rounded-md border border-gray-800 bg-[#0d1117]">
        <Table>
          <TableHeader className="bg-[#0d1117]">
            <TableRow className="border-gray-800">
              <TableHead>Transaction ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Timestamp</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Block</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {transactions.map((tx) => (
              <TableRow key={tx.id} className="border-gray-800">
                <TableCell className="font-mono text-xs">{tx.id}</TableCell>
                <TableCell>
                  {tx.type === "Threat Record"
                    ? "Threat Record"
                    : tx.type === "Model Update"
                      ? "Model Update"
                      : tx.type === "System Config"
                        ? "System Config"
                        : tx.type}
                </TableCell>
                <TableCell>{tx.timestamp}</TableCell>
                <TableCell>
                  <Badge
                    variant={
                      tx.status === "Confirmed" ? "secondary" : tx.status === "Pending" ? "default" : "destructive"
                    }
                    className={
                      tx.status === "Confirmed"
                        ? "bg-blue-900 text-blue-100"
                        : tx.status === "Pending"
                          ? "bg-blue-500 text-white"
                          : ""
                    }
                  >
                    {tx.status}
                  </Badge>
                </TableCell>
                <TableCell>{tx.status === "Pending" ? "Pending" : tx.block}</TableCell>
                <TableCell>
                  <Button variant="ghost" size="sm">
                    View
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
