"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Search, Filter } from "lucide-react"

export function ThreatsFilters() {
  const [searchQuery, setSearchQuery] = useState("")
  const [severity, setSeverity] = useState("all")
  const [status, setStatus] = useState("all")

  return (
    <div className="flex flex-col space-y-4 md:flex-row md:items-end md:space-x-4 md:space-y-0">
      <div className="flex-1 space-y-2">
        <label htmlFor="search" className="text-sm font-medium">
          Search Threats
        </label>
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            id="search"
            type="search"
            placeholder="Search by ID, name, or source..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      <div className="w-full space-y-2 md:w-40">
        <label htmlFor="severity" className="text-sm font-medium">
          Severity
        </label>
        <Select value={severity} onValueChange={setSeverity}>
          <SelectTrigger id="severity">
            <SelectValue placeholder="Select severity" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Severities</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="low">Low</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="w-full space-y-2 md:w-40">
        <label htmlFor="status" className="text-sm font-medium">
          Status
        </label>
        <Select value={status} onValueChange={setStatus}>
          <SelectTrigger id="status">
            <SelectValue placeholder="Select status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Statuses</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="investigating">Investigating</SelectItem>
            <SelectItem value="resolved">Resolved</SelectItem>
            <SelectItem value="false-positive">False Positive</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <Button variant="outline" className="w-full md:w-auto">
        <Filter className="mr-2 h-4 w-4" />
        More Filters
      </Button>
    </div>
  )
}
