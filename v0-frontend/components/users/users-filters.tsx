"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Search, Filter } from "lucide-react"

export function UsersFilters() {
  const [searchQuery, setSearchQuery] = useState("")
  const [role, setRole] = useState("all")
  const [status, setStatus] = useState("all")

  return (
    <div className="flex flex-col space-y-4 md:flex-row md:items-end md:space-x-4 md:space-y-0">
      <div className="flex-1 space-y-2">
        <label htmlFor="search" className="text-sm font-medium">
          Search Users
        </label>
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            id="search"
            type="search"
            placeholder="Search by name, email, or role..."
            className="pl-8 bg-[#1a1d29] border-gray-700"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      <div className="w-full space-y-2 md:w-40">
        <label htmlFor="role" className="text-sm font-medium">
          Role
        </label>
        <Select value={role} onValueChange={setRole}>
          <SelectTrigger id="role" className="bg-[#1a1d29] border-gray-700">
            <SelectValue placeholder="Select role" />
          </SelectTrigger>
          <SelectContent className="bg-[#1a1d29] border-gray-700">
            <SelectItem value="all">All Roles</SelectItem>
            <SelectItem value="admin">Admin</SelectItem>
            <SelectItem value="analyst">Analyst</SelectItem>
            <SelectItem value="user">User</SelectItem>
            <SelectItem value="guest">Guest</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="w-full space-y-2 md:w-40">
        <label htmlFor="status" className="text-sm font-medium">
          Status
        </label>
        <Select value={status} onValueChange={setStatus}>
          <SelectTrigger id="status" className="bg-[#1a1d29] border-gray-700">
            <SelectValue placeholder="Select status" />
          </SelectTrigger>
          <SelectContent className="bg-[#1a1d29] border-gray-700">
            <SelectItem value="all">All Statuses</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
            <SelectItem value="locked">Locked</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <Button variant="outline" className="w-full md:w-auto border-gray-700">
        <Filter className="mr-2 h-4 w-4" />
        More Filters
      </Button>
    </div>
  )
}
