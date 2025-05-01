import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export function UsersListSkeleton() {
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
          {Array(5)
            .fill(0)
            .map((_, i) => (
              <TableRow key={i} className="border-gray-800">
                <TableCell>
                  <div className="flex items-center gap-3">
                    <Skeleton className="h-10 w-10 rounded-full bg-gray-700" />
                    <Skeleton className="h-4 w-32 bg-gray-700" />
                  </div>
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-40 bg-gray-700" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-5 w-16 rounded-full bg-gray-700" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-5 w-16 rounded-full bg-gray-700" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-24 bg-gray-700" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-8 w-16 rounded-md bg-gray-700" />
                </TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </div>
  )
}
