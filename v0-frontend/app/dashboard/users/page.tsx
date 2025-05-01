import { Suspense } from "react"
import { UsersHeader } from "@/components/users/users-header"
import { UsersList } from "@/components/users/users-list"
import { UsersFilters } from "@/components/users/users-filters"
import { UsersListSkeleton } from "@/components/users/users-list-skeleton"

export default function UsersPage() {
  return (
    <div className="space-y-6 ml-64">
      <UsersHeader />
      <UsersFilters />
      <Suspense fallback={<UsersListSkeleton />}>
        <UsersList />
      </Suspense>
    </div>
  )
}
