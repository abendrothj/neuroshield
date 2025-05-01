import { Button } from "@/components/ui/button"
import { PlusCircle } from "lucide-react"

export function UsersHeader() {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">Users</h1>
        <p className="text-muted-foreground">Manage user accounts and permissions</p>
      </div>
      <Button className="bg-blue-500 hover:bg-blue-600">
        <PlusCircle className="mr-2 h-4 w-4" />
        Add User
      </Button>
    </div>
  )
}
