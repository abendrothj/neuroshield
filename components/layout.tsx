"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { usePathname } from "next/navigation"
import { LayoutDashboard, FileText, Settings, Shield, Menu, X, Bell, User, ChevronDown } from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import SocketClient from "@/lib/socket"

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const pathname = usePathname()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [notifications, setNotifications] = useState<string[]>([])
  const socket = SocketClient.getInstance()

  useEffect(() => {
    // Connect to WebSocket server
    socket.connect()

    // Subscribe to various events
    socket.subscribe("threatDetected", (data) => {
      setNotifications((prev) => [`Threat detected: ${data.type}`, ...prev.slice(0, 4)])
    })

    // Update connection status
    const checkConnection = () => {
      setIsConnected(socket.isSocketConnected())
    }

    const interval = setInterval(checkConnection, 5000)

    // Cleanup on unmount
    return () => {
      socket.unsubscribe("threatDetected", () => {})
      clearInterval(interval)
      socket.disconnect()
    }
  }, [])

  const navigation = [
    { name: "Dashboard", href: "/", icon: LayoutDashboard },
    { name: "Logs", href: "/logs", icon: FileText },
    { name: "Settings", href: "/settings", icon: Settings },
  ]

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar for desktop */}
      <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
        <div className="flex flex-col flex-grow glass-effect">
          <div className="flex items-center h-16 flex-shrink-0 px-4 border-b border-white/10">
            <Shield className="h-8 w-8 text-primary glow" />
            <span className="ml-2 text-xl font-bold glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              NeuraShield
            </span>
          </div>
          <div className="flex-1 flex flex-col overflow-y-auto pt-5 pb-4">
            <nav className="mt-5 flex-1 px-2 space-y-1">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    pathname === item.href ? "bg-muted text-primary glow" : "text-muted-foreground hover:bg-muted/50",
                    "group flex items-center px-3 py-3 text-sm font-medium rounded-md transition-all duration-200",
                  )}
                >
                  <item.icon
                    className={cn(
                      pathname === item.href ? "text-primary" : "text-muted-foreground group-hover:text-primary",
                      "mr-3 flex-shrink-0 h-5 w-5 transition-colors",
                    )}
                  />
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
          <div className="p-4 border-t border-white/10">
            <div className="flex items-center">
              <div className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"} mr-2`} />
              <span className="text-sm text-muted-foreground">{isConnected ? "Connected" : "Disconnected"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile sidebar */}
      <div
        className={`fixed inset-0 z-40 md:hidden transition-opacity duration-300 ${
          isSidebarOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm" onClick={() => setIsSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 w-full max-w-xs glass-effect">
          <div className="flex items-center justify-between h-16 px-4 border-b border-white/10">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-primary" />
              <span className="ml-2 text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                NeuraShield
              </span>
            </div>
            <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(false)}>
              <X className="h-6 w-6" />
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto pt-5 pb-4">
            <nav className="mt-5 px-2 space-y-1">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    pathname === item.href ? "bg-muted text-primary" : "text-muted-foreground hover:bg-muted/50",
                    "group flex items-center px-3 py-3 text-sm font-medium rounded-md",
                  )}
                  onClick={() => setIsSidebarOpen(false)}
                >
                  <item.icon
                    className={cn(
                      pathname === item.href ? "text-primary" : "text-muted-foreground",
                      "mr-3 flex-shrink-0 h-5 w-5",
                    )}
                  />
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="md:pl-64 flex flex-col flex-1">
        <div className="sticky top-0 z-10 flex-shrink-0 h-16 glass-effect border-b border-white/10">
          <div className="flex items-center justify-between px-4 h-full">
            <div className="flex items-center">
              <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsSidebarOpen(true)}>
                <Menu className="h-6 w-6" />
              </Button>
              <h1 className="ml-2 md:ml-0 text-lg font-semibold">
                {navigation.find((item) => item.href === pathname)?.name || "Dashboard"}
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="relative">
                    <Bell className="h-5 w-5" />
                    {notifications.length > 0 && (
                      <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-primary" />
                    )}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-80">
                  <DropdownMenuLabel>Notifications</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {notifications.length > 0 ? (
                    notifications.map((notification, i) => (
                      <DropdownMenuItem key={i} className="py-2">
                        {notification}
                      </DropdownMenuItem>
                    ))
                  ) : (
                    <DropdownMenuItem disabled>No new notifications</DropdownMenuItem>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center">
                      <User className="h-4 w-4" />
                    </div>
                    <span className="hidden md:inline-block">Admin</span>
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>Profile</DropdownMenuItem>
                  <DropdownMenuItem>Account Settings</DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>Log out</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>
        <main className="flex-1 overflow-y-auto bg-gradient-to-b from-background to-background/80">
          <div className="py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">{children}</div>
          </div>
        </main>
      </div>
    </div>
  )
}

