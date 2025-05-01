"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { SecurityEvent } from "@/src/types/blockchain";
import { getBlockchainService } from "@/src/lib/api/service-provider";
import { Skeleton } from "@/components/ui/skeleton";

export function BlockchainEvents() {
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [blockchainStatus, setBlockchainStatus] = useState<string>("unknown");
  
  const blockchainService = getBlockchainService();

  const fetchEvents = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await blockchainService.getEvents();
      setEvents(data);
    } catch (err) {
      setError("Failed to fetch blockchain events");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const checkBlockchainStatus = async () => {
    try {
      const status = await blockchainService.checkBlockchainStatus();
      setBlockchainStatus(status.status);
    } catch (err) {
      setBlockchainStatus("error");
      console.error(err);
    }
  };

  useEffect(() => {
    fetchEvents();
    checkBlockchainStatus();
    const interval = setInterval(checkBlockchainStatus, 30000); // Check status every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getEventTypeBadge = (type: string) => {
    const badgeVariant = {
      "INFO": "secondary",
      "WARNING": "warning",
      "CRITICAL": "destructive",
      "ERROR": "outline"
    }[type] || "default";
    
    return <Badge variant={badgeVariant}>{type}</Badge>;
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDetails = (details: string) => {
    try {
      const parsedDetails = JSON.parse(details);
      return parsedDetails.message || JSON.stringify(parsedDetails).substring(0, 100);
    } catch (error) {
      return details.substring(0, 100);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Blockchain Security Events</CardTitle>
            <CardDescription>
              Immutable records of security events on the blockchain
            </CardDescription>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant={blockchainStatus === "ok" ? "default" : "destructive"}>
              Blockchain: {blockchainStatus}
            </Badge>
            <Button onClick={fetchEvents} disabled={isLoading}>
              {isLoading ? "Loading..." : "Refresh"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Event ID</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Details</TableHead>
                <TableHead>IPFS</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {events.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-4">
                    No events found
                  </TableCell>
                </TableRow>
              ) : (
                events.map((event) => (
                  <TableRow key={event.id}>
                    <TableCell className="font-mono text-xs">{event.id}</TableCell>
                    <TableCell>{formatDate(event.timestamp)}</TableCell>
                    <TableCell>{getEventTypeBadge(event.type)}</TableCell>
                    <TableCell>
                      <div className="max-w-sm truncate">{formatDetails(event.details)}</div>
                    </TableCell>
                    <TableCell>
                      {event.ipfsHash ? (
                        <a 
                          href={`${process.env.NEXT_PUBLIC_IPFS_GATEWAY}${event.ipfsHash}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:text-blue-700 underline"
                        >
                          View Details
                        </a>
                      ) : (
                        "N/A"
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
} 