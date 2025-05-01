import { Button } from "@/components/ui/button"
import { PlusCircle } from "lucide-react"

export function AIModelsHeader() {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">AI Models</h1>
        <p className="text-muted-foreground">Manage and monitor AI model performance</p>
      </div>
      <Button>
        <PlusCircle className="mr-2 h-4 w-4" />
        Deploy New Model
      </Button>
    </div>
  )
}
