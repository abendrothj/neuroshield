import { Skeleton } from "@/components/ui/skeleton"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export function BlockchainSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {Array(4)
          .fill(0)
          .map((_, i) => (
            <Card key={i} className="bg-[#0d1117] border-gray-800">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <Skeleton className="h-4 w-24 bg-gray-700" />
                <Skeleton className="h-4 w-4 bg-gray-700" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-20 mb-1 bg-gray-700" />
                <Skeleton className="h-3 w-32 bg-gray-700" />
              </CardContent>
            </Card>
          ))}
      </div>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-40 bg-gray-700" />
          <Skeleton className="h-9 w-20 bg-gray-700" />
        </div>
        <div className="rounded-md border border-gray-800 bg-[#0d1117]">
          <Table>
            <TableHeader>
              <TableRow className="border-gray-800">
                <TableHead>Transaction ID</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Block</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Array(5)
                .fill(0)
                .map((_, i) => (
                  <TableRow key={i} className="border-gray-800">
                    <TableCell>
                      <Skeleton className="h-4 w-32 bg-gray-700" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-20 bg-gray-700" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-24 bg-gray-700" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-5 w-16 rounded-full bg-gray-700" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-12 bg-gray-700" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-8 w-16 rounded-md bg-gray-700" />
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}
