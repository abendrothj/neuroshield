"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, RefreshCw } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/auth/auth-provider"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { useAIModels } from "@/hooks/api/use-ai-models-data"

export function AIModelsList() {
  const { models, loading, error, startRetraining } = useAIModels()
  const [retrainingId, setRetrainingId] = useState(null)
  const [selectedModel, setSelectedModel] = useState(null)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const { toast } = useToast()
  const { user } = useAuth()

  const isAdmin = user?.role === "admin"

  const handleRetrain = async (id) => {
    if (!isAdmin) {
      toast({
        title: "Permission denied",
        description: "Only administrators can retrain models",
        variant: "destructive",
      })
      return
    }

    try {
      setRetrainingId(id)
      const result = await startRetraining(id)

      if (result.success) {
        toast({
          title: "Retraining started",
          description: result.message || "Model retraining has been initiated",
        })
      } else {
        toast({
          title: "Retraining failed",
          description: result.message || "Failed to start model retraining",
          variant: "destructive",
        })
      }
    } catch (err) {
      console.error("Failed to retrain model:", err)
      toast({
        title: "Retraining failed",
        description: "An error occurred while trying to retrain the model",
        variant: "destructive",
      })
    } finally {
      setRetrainingId(null)
    }
  }

  const handleViewDetails = (model) => {
    setSelectedModel(model)
    setDetailsOpen(true)
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
          <h2 className="text-xl font-semibold">Deployed Models</h2>
          <Button variant="outline" size="sm">
            View All
          </Button>
        </div>
        <div className="rounded-md border border-gray-800 bg-[#0d1117]">
          <Table>
            <TableHeader className="bg-[#0d1117]">
              <TableRow className="border-gray-800">
                <TableHead>Model Name</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Last Updated</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                    No models found
                  </TableCell>
                </TableRow>
              ) : (
                models.map((model) => (
                  <TableRow key={model.id} className="border-gray-800">
                    <TableCell className="font-medium">{model.name}</TableCell>
                    <TableCell>{model.version}</TableCell>
                    <TableCell>{model.type}</TableCell>
                    <TableCell>{model.accuracy}%</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          model.status === "Active" ? "secondary" : model.status === "Training" ? "default" : "outline"
                        }
                        className={
                          model.status === "Active"
                            ? "bg-green-900/20 text-green-400 border-green-800"
                            : model.status === "Training"
                              ? "bg-blue-900/20 text-blue-400 border-blue-800"
                              : "bg-gray-900/20 text-gray-400 border-gray-800"
                        }
                      >
                        {model.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{model.lastUpdated}</TableCell>
                    <TableCell>
                      <div className="flex space-x-2">
                        <Button variant="ghost" size="sm" onClick={() => handleViewDetails(model)}>
                          View
                        </Button>
                        {isAdmin && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleRetrain(model.id)}
                            disabled={retrainingId === model.id || model.status === "Training"}
                          >
                            {retrainingId === model.id ? (
                              <Loader2 className="h-4 w-4 animate-spin mr-1" />
                            ) : (
                              <RefreshCw className="h-4 w-4 mr-1" />
                            )}
                            Retrain
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="bg-[#0d1117] border-gray-800 text-white">
          <DialogHeader>
            <DialogTitle>Model Details</DialogTitle>
            <DialogDescription>Detailed information about the selected AI model</DialogDescription>
          </DialogHeader>

          {selectedModel && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Name</h4>
                  <p>{selectedModel.name}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Version</h4>
                  <p>{selectedModel.version}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Type</h4>
                  <p>{selectedModel.type}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Accuracy</h4>
                  <p>{selectedModel.accuracy}%</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Status</h4>
                  <Badge
                    variant={
                      selectedModel.status === "Active"
                        ? "secondary"
                        : selectedModel.status === "Training"
                          ? "default"
                          : "outline"
                    }
                    className={
                      selectedModel.status === "Active"
                        ? "bg-green-900/20 text-green-400 border-green-800"
                        : selectedModel.status === "Training"
                          ? "bg-blue-900/20 text-blue-400 border-blue-800"
                          : "bg-gray-900/20 text-gray-400 border-gray-800"
                    }
                  >
                    {selectedModel.status}
                  </Badge>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Last Updated</h4>
                  <p>{selectedModel.lastUpdated}</p>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-1">Performance Metrics</h4>
                <div className="p-4 bg-[#1a1d29] rounded-md border border-gray-800">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-muted-foreground">Precision</p>
                      <p className="text-sm">{(selectedModel.accuracy * 0.98).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Recall</p>
                      <p className="text-sm">{(selectedModel.accuracy * 0.96).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">F1 Score</p>
                      <p className="text-sm">{(selectedModel.accuracy * 0.97).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Training Time</p>
                      <p className="text-sm">4.2 hours</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-end space-x-2 pt-4">
                {isAdmin && selectedModel.status !== "Training" && (
                  <Button
                    variant="outline"
                    onClick={() => {
                      handleRetrain(selectedModel.id)
                      setDetailsOpen(false)
                    }}
                  >
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Retrain Model
                  </Button>
                )}
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
