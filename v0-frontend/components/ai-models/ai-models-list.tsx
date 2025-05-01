import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getAIModels } from "@/lib/api/ai-models"

export async function AIModelsList() {
  const models = await getAIModels()

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Deployed Models</h2>
        <Button variant="outline" size="sm">
          View All
        </Button>
      </div>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
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
            {models.map((model) => (
              <TableRow key={model.id}>
                <TableCell className="font-medium">{model.name}</TableCell>
                <TableCell>{model.version}</TableCell>
                <TableCell>{model.type}</TableCell>
                <TableCell>{model.accuracy}%</TableCell>
                <TableCell>
                  <Badge
                    variant={
                      model.status === "Active" ? "secondary" : model.status === "Training" ? "default" : "outline"
                    }
                  >
                    {model.status}
                  </Badge>
                </TableCell>
                <TableCell>{model.lastUpdated}</TableCell>
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
