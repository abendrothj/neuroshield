"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Loader2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useThreats, useThreatDetails } from "@/hooks/api/use-threat-data"

export function ThreatsList({ severity = "all", status = "all", searchQuery = "" }) {
  const { threats, loading, error, updateThreat } = useThreats(severity, status, searchQuery)
  const [selectedThreatId, setSelectedThreatId] = useState<string | null>(null)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const { threat: selectedThreat, loading: detailsLoading, fetchThreat } = useThreatDetails()
  const { toast } = useToast()

  const handleViewDetails = async (id: string) => {
    try {
      setSelectedThreatId(id)
      await fetchThreat(id)
      setDetailsOpen(true)
    } catch (err) {
      console.error("Failed to fetch threat details:", err)
      toast({
        title: "Error",
        description: "Failed to load threat details",
        variant: "destructive",
      })
    }
  }

  const handleStatusChange = async (id: string, newStatus: string) => {
    try {
      await updateThreat(id, newStatus)
      toast({
        title: "Status updated",
        description: `Threat status changed to ${newStatus}`,
      })
    } catch (err) {
      console.error("Failed to update threat status:", err)
      toast({
        title: "Error",
        description: "Failed to update threat status",
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
      <div className="rounded-md border border-gray-800 bg-[#0d1117]">
        <Table>
          <TableHeader className="bg-[#0d1117]">
            <TableRow className="border-gray-800">
              <TableHead>ID</TableHead>
              <TableHead>Threat</TableHead>
              <TableHead>Source</TableHead>
              <TableHead>Severity</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Detected</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {threats.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                  No threats found
                </TableCell>
              </TableRow>
            ) : (
              threats.map((threat) => (
                <TableRow key={threat.id} className="border-gray-800">
                  <TableCell className="font-mono text-xs">{threat.id}</TableCell>
                  <TableCell className="font-medium">{threat.name}</TableCell>
                  <TableCell>{threat.source}</TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        threat.severity === "Critical"
                          ? "destructive"
                          : threat.severity === "High"
                            ? "default"
                            : threat.severity === "Medium"
                              ? "secondary"
                              : "outline"
                      }
                      className={
                        threat.severity === "Critical"
                          ? "bg-red-900/20 text-red-400 border-red-800"
                          : threat.severity === "High"
                            ? "bg-orange-900/20 text-orange-400 border-orange-800"
                            : threat.severity === "Medium"
                              ? "bg-yellow-900/20 text-yellow-400 border-yellow-800"
                              : "bg-gray-900/20 text-gray-400 border-gray-800"
                      }
                    >
                      {threat.severity}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        threat.status === "Active"
                          ? "destructive"
                          : threat.status === "Investigating"
                            ? "default"
                            : threat.status === "Resolved"
                              ? "secondary"
                              : "outline"
                      }
                      className={
                        threat.status === "Active"
                          ? "bg-red-900/20 text-red-400 border-red-800"
                          : threat.status === "Investigating"
                            ? "bg-blue-900/20 text-blue-400 border-blue-800"
                            : threat.status === "Resolved"
                              ? "bg-green-900/20 text-green-400 border-green-800"
                              : "bg-gray-900/20 text-gray-400 border-gray-800"
                      }
                    >
                      {threat.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{threat.detectedAt}</TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleViewDetails(threat.id)}
                      disabled={detailsLoading && selectedThreatId === threat.id}
                    >
                      {detailsLoading && selectedThreatId === threat.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        "View"
                      )}
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="bg-[#0d1117] border-gray-800 text-white max-w-3xl">
          <DialogHeader>
            <DialogTitle>Threat Details</DialogTitle>
            <DialogDescription>Detailed information about the selected threat</DialogDescription>
          </DialogHeader>

          {selectedThreat && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">ID</h4>
                  <p className="font-mono">{selectedThreat.id}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Detected At</h4>
                  <p>{selectedThreat.detectedAt}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Name</h4>
                  <p>{selectedThreat.name}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Source</h4>
                  <p>{selectedThreat.source}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Severity</h4>
                  <Badge
                    variant={
                      selectedThreat.severity === "Critical"
                        ? "destructive"
                        : selectedThreat.severity === "High"
                          ? "default"
                          : selectedThreat.severity === "Medium"
                            ? "secondary"
                            : "outline"
                    }
                    className={
                      selectedThreat.severity === "Critical"
                        ? "bg-red-900/20 text-red-400 border-red-800"
                        : selectedThreat.severity === "High"
                          ? "bg-orange-900/20 text-orange-400 border-orange-800"
                          : selectedThreat.severity === "Medium"
                            ? "bg-yellow-900/20 text-yellow-400 border-yellow-800"
                            : "bg-gray-900/20 text-gray-400 border-gray-800"
                    }
                  >
                    {selectedThreat.severity}
                  </Badge>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Status</h4>
                  <div className="flex items-center space-x-2">
                    <Select
                      defaultValue={selectedThreat.status}
                      onValueChange={(value) => handleStatusChange(selectedThreat.id, value)}
                    >
                      <SelectTrigger className="w-[180px] bg-[#1a1d29] border-gray-700">
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1d29] border-gray-700">
                        <SelectItem value="Active">Active</SelectItem>
                        <SelectItem value="Investigating">Investigating</SelectItem>
                        <SelectItem value="Resolved">Resolved</SelectItem>
                        <SelectItem value="False Positive">False Positive</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-1">Description</h4>
                <div className="p-4 bg-[#1a1d29] rounded-md border border-gray-800">
                  <p>{selectedThreat.description}</p>
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
