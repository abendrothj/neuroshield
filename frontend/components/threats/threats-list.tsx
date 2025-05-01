import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getThreats } from "@/lib/api/threats"

export async function ThreatsList() {
  const threats = await getThreats()

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
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
          {threats.map((threat) => (
            <TableRow key={threat.id}>
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
                >
                  {threat.status}
                </Badge>
              </TableCell>
              <TableCell>{threat.detectedAt}</TableCell>
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
  )
}
