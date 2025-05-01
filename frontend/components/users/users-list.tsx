import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { getUsers } from "@/lib/api/users"

export async function UsersList() {
  const users = await getUsers()

  return (
    <div className="rounded-md border border-gray-800 bg-[#0d1117]">
      <Table>
        <TableHeader className="bg-[#0d1117]">
          <TableRow className="border-gray-800">
            <TableHead>User</TableHead>
            <TableHead>Email</TableHead>
            <TableHead>Role</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Last Active</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {users.map((user) => (
            <TableRow key={user.id} className="border-gray-800">
              <TableCell>
                <div className="flex items-center gap-3">
                  <Avatar>
                    <AvatarImage src={user.avatar || "/placeholder.svg"} alt={user.name} />
                    <AvatarFallback>{user.name.substring(0, 2).toUpperCase()}</AvatarFallback>
                  </Avatar>
                  <div className="font-medium">{user.name}</div>
                </div>
              </TableCell>
              <TableCell>{user.email}</TableCell>
              <TableCell>
                <Badge
                  variant="outline"
                  className={
                    user.role === "Admin"
                      ? "bg-red-900/20 text-red-400 border-red-800"
                      : user.role === "Analyst"
                        ? "bg-blue-900/20 text-blue-400 border-blue-800"
                        : "bg-gray-900/20 text-gray-400 border-gray-800"
                  }
                >
                  {user.role}
                </Badge>
              </TableCell>
              <TableCell>
                <Badge
                  variant={
                    user.status === "Active" ? "secondary" : user.status === "Inactive" ? "outline" : "destructive"
                  }
                  className={
                    user.status === "Active"
                      ? "bg-green-900/20 text-green-400 border-green-800"
                      : user.status === "Inactive"
                        ? "bg-gray-900/20 text-gray-400 border-gray-800"
                        : "bg-red-900/20 text-red-400 border-red-800"
                  }
                >
                  {user.status}
                </Badge>
              </TableCell>
              <TableCell>{user.lastActive}</TableCell>
              <TableCell>
                <Button variant="ghost" size="sm">
                  Edit
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
