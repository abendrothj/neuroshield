"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ChevronDown, ChevronUp, Copy, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

// Generate dummy blockchain logs
const generateLogs = () => [
  {
    id: 1,
    timestamp: "2025-03-31 14:32",
    event: "Threat Blocked",
    details: "IP 192.168.1.1 isolated",
    blockHash: "0xabc123def456789abcdef0123456789abcdef012",
  },
  {
    id: 2,
    timestamp: "2025-03-31 13:45",
    event: "AI Prediction",
    details: "Potential DDoS attack predicted",
    blockHash: "0x789abc123def456789abcdef0123456789abcdef",
  },
  {
    id: 3,
    timestamp: "2025-03-31 12:18",
    event: "System Update",
    details: "Security patches applied",
    blockHash: "0xdef456789abcdef0123456789abcdef012345678",
  },
  {
    id: 4,
    timestamp: "2025-03-31 10:05",
    event: "Threat Blocked",
    details: "Malware signature detected",
    blockHash: "0x456789abcdef0123456789abcdef0123456789ab",
  },
  {
    id: 5,
    timestamp: "2025-03-31 09:22",
    event: "Authentication",
    details: "Admin login from trusted device",
    blockHash: "0x123456789abcdef0123456789abcdef0123456789",
  },
]

export default function BlockchainLogs() {
  const [logs] = useState(generateLogs())
  const [sortField, setSortField] = useState("timestamp")
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc")
  const [searchTerm, setSearchTerm] = useState("")

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  const sortedLogs = [...logs]
    .filter(
      (log) =>
        log.event.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.details.toLowerCase().includes(searchTerm.toLowerCase()),
    )
    .sort((a, b) => {
      if (sortDirection === "asc") {
        return a[sortField as keyof typeof a] > b[sortField as keyof typeof b] ? 1 : -1
      } else {
        return a[sortField as keyof typeof a] < b[sortField as keyof typeof b] ? 1 : -1
      }
    })

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <Card className="glass-effect">
      <CardHeader className="pb-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <CardTitle className="text-xl">Blockchain Logs</CardTitle>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search logs..."
              className="pl-8 bg-muted/50 border-white/10 focus-visible:ring-primary"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-white/10 overflow-hidden">
          <div className="max-h-[400px] overflow-y-auto">
            <Table>
              <TableHeader className="bg-muted/30 sticky top-0 z-10">
                <TableRow>
                  <TableHead
                    className="text-muted-foreground cursor-pointer hover:text-primary"
                    onClick={() => handleSort("timestamp")}
                  >
                    <div className="flex items-center">
                      Timestamp
                      {sortField === "timestamp" &&
                        (sortDirection === "asc" ? (
                          <ChevronUp className="ml-1 h-4 w-4" />
                        ) : (
                          <ChevronDown className="ml-1 h-4 w-4" />
                        ))}
                    </div>
                  </TableHead>
                  <TableHead
                    className="text-muted-foreground cursor-pointer hover:text-primary"
                    onClick={() => handleSort("event")}
                  >
                    <div className="flex items-center">
                      Event
                      {sortField === "event" &&
                        (sortDirection === "asc" ? (
                          <ChevronUp className="ml-1 h-4 w-4" />
                        ) : (
                          <ChevronDown className="ml-1 h-4 w-4" />
                        ))}
                    </div>
                  </TableHead>
                  <TableHead className="text-muted-foreground">Details</TableHead>
                  <TableHead className="text-muted-foreground">Block Hash</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedLogs.map((log) => (
                  <TableRow key={log.id} className="hover:bg-primary/10 transition-colors duration-150">
                    <TableCell className="font-mono text-sm">{log.timestamp}</TableCell>
                    <TableCell>
                      <span
                        className={`px-2 py-1 rounded-full text-xs ${
                          log.event === "Threat Blocked"
                            ? "bg-red-900/30 text-red-400"
                            : log.event === "AI Prediction"
                              ? "bg-secondary/30 text-secondary"
                              : "bg-blue-900/30 text-blue-400"
                        }`}
                      >
                        {log.event}
                      </span>
                    </TableCell>
                    <TableCell>{log.details}</TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <span className="font-mono text-xs truncate max-w-[120px] md:max-w-[200px]">
                          {log.blockHash}
                        </span>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="ml-1 h-6 w-6 text-muted-foreground hover:text-primary"
                          onClick={() => copyToClipboard(log.blockHash)}
                        >
                          <Copy className="h-3 w-3" />
                          <span className="sr-only">Copy hash</span>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

