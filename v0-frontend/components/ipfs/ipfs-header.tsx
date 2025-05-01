import { Button } from "@/components/ui/button"
import { Upload } from "lucide-react"

export function IPFSHeader() {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">IPFS Storage</h1>
        <p className="text-muted-foreground">Manage and monitor IPFS storage</p>
      </div>
      <Button>
        <Upload className="mr-2 h-4 w-4" />
        Upload File
      </Button>
    </div>
  )
}
