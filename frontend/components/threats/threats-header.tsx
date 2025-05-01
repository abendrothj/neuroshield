import { Button } from "@/components/ui/button"
import { PlusCircle } from "lucide-react"

export function ThreatsHeader() {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">Threat Management</h1>
        <p className="text-muted-foreground">Monitor and manage detected threats across your network</p>
      </div>
      <Button>
        <PlusCircle className="mr-2 h-4 w-4" />
        Add Custom Rule
      </Button>
    </div>
  )
}
