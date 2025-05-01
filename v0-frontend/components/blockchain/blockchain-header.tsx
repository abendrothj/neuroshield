import { Button } from "@/components/ui/button"
import { PlusCircle } from "lucide-react"

export function BlockchainHeader() {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">Blockchain</h1>
        <p className="text-muted-foreground">Monitor blockchain transactions and network status</p>
      </div>
      <Button className="bg-blue-500 hover:bg-blue-600">
        <PlusCircle className="mr-2 h-4 w-4" />
        New Transaction
      </Button>
    </div>
  )
}
