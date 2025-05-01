"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, Download } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { useToast } from "@/hooks/use-toast"
import { useBlockchainTransactions } from "@/hooks/api/use-blockchain-data"

export function BlockchainTransactions() {
  const { transactions, loading, error, downloadTransactions } = useBlockchainTransactions()
  const [selectedTx, setSelectedTx] = useState(null)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const { toast } = useToast()

  const handleViewDetails = (tx) => {
    setSelectedTx(tx)
    setDetailsOpen(true)
  }

  const handleDownload = () => {
    try {
      downloadTransactions()
      toast({
        title: "Download started",
        description: "Your transaction log is being downloaded",
      })
    } catch (err) {
      console.error("Failed to download transactions:", err)
      toast({
        title: "Download failed",
        description: "Failed to download transaction log",
        variant: "destructive",
      })
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-900/20 p-4 border border-red-800">
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  return (
    <>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Recent Transactions</h2>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="mr-2 h-4 w-4" />
            Download JSON
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
              {transactions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    No transactions found
                  </TableCell>
                </TableRow>
              ) : (
                transactions.map((tx) => (
                  <TableRow
                    key={tx.id}
                    className={`border-gray-800 ${tx.type === "Threat Record" ? "bg-red-900/10" : ""}`}
                  >
                    <TableCell className="font-mono text-xs">{tx.id}</TableCell>
                    <TableCell className={tx.type === "Threat Record" ? "text-red-400 font-medium" : ""}>
                      {tx.type}
                    </TableCell>
                    <TableCell>{tx.timestamp}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          tx.status === "Confirmed" ? "secondary" : tx.status === "Pending" ? "default" : "destructive"
                        }
                        className={
                          tx.status === "Confirmed"
                            ? "bg-blue-900/20 text-blue-100"
                            : tx.status === "Pending"
                              ? "bg-blue-500 text-white"
                              : "bg-red-900/20 text-red-400"
                        }
                      >
                        {tx.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{tx.status === "Pending" ? "Pending" : tx.block}</TableCell>
                    <TableCell>
                      <Button variant="ghost" size="sm" onClick={() => handleViewDetails(tx)}>
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="bg-[#0d1117 border-gray-800 text-white">
          <DialogHeader>
            <DialogTitle>Transaction Details</DialogTitle>
            <DialogDescription>Detailed information about the selected blockchain transaction</DialogDescription>
          </DialogHeader>

          {selectedTx && (
            <div className="space-y-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground">Transaction ID</h4>
                <p className="font-mono text-xs bg-[#1a1d29] p-2 rounded-md border border-gray-800 break-all">
                  {selectedTx.id}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Type</h4>
                  <p className={selectedTx.type === "Threat Record" ? "text-red-400" : ""}>{selectedTx.type}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Timestamp</h4>
                  <p>{selectedTx.timestamp}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Status</h4>
                  <Badge
                    variant={
                      selectedTx.status === "Confirmed"
                        ? "secondary"
                        : selectedTx.status === "Pending"
                          ? "default"
                          : "destructive"
                    }
                    className={
                      selectedTx.status === "Confirmed"
                        ? "bg-blue-900/20 text-blue-100"
                        : selectedTx.status === "Pending"
                          ? "bg-blue-500 text-white"
                          : "bg-red-900/20 text-red-400"
                    }
                  >
                    {selectedTx.status}
                  </Badge>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Block</h4>
                  <p>{selectedTx.status === "Pending" ? "Pending" : selectedTx.block}</p>
                </div>
              </div>

              <div className="flex justify-end space-x-2 pt-4">
                <Button variant="outline" onClick={() => setDetailsOpen(false)}>
                  Close
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
