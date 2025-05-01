import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getIPFSFiles } from "@/lib/api/ipfs"

export async function IPFSFiles() {
  const files = await getIPFSFiles()

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Recent Files</h2>
        <Button variant="outline" size="sm">
          View All
        </Button>
      </div>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>CID</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Uploaded</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {files.map((file) => (
              <TableRow key={file.id}>
                <TableCell className="font-medium">{file.name}</TableCell>
                <TableCell className="font-mono text-xs">{file.cid}</TableCell>
                <TableCell>{file.size}</TableCell>
                <TableCell>{file.type}</TableCell>
                <TableCell>
                  <Badge
                    variant={
                      file.status === "Pinned" ? "secondary" : file.status === "Uploading" ? "default" : "outline"
                    }
                  >
                    {file.status}
                  </Badge>
                </TableCell>
                <TableCell>{file.uploaded}</TableCell>
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
