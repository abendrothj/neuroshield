"use client"

import { usePathname } from "next/navigation"
import Link from "next/link"
import { BarChart3, Shield, Database, Brain, HardDrive, Settings, Users, Bell, LogOut } from "lucide-react"
import { useAuth } from "@/lib/auth/auth-provider"

export function DashboardSidebar() {
  const pathname = usePathname()
  const { logout, user } = useAuth()

  const isActive = (path: string) => {
    return pathname === path || pathname.startsWith(`${path}/`)
  }

  const menuItems = [
    {
      title: "Dashboard",
      icon: BarChart3,
      href: "/dashboard",
    },
    {
      title: "Threats",
      icon: Shield,
      href: "/dashboard/threats",
    },
    {
      title: "Blockchain",
      icon: Database,
      href: "/dashboard/blockchain",
    },
    {
      title: "AI Models",
      icon: Brain,
      href: "/dashboard/ai-models",
    },
    {
      title: "IPFS Storage",
      icon: HardDrive,
      href: "/dashboard/ipfs",
    },
    {
      title: "Settings",
      icon: Settings,
      href: "/dashboard/settings",
    },
    {
      title: "Users",
      icon: Users,
      href: "/dashboard/users",
    },
  ]

  return (
    <div className="fixed inset-y-0 left-0 z-50 w-64 bg-[#0a0c10] border-r border-gray-800">
      <div className="flex h-16 items-center px-4 border-b border-gray-800">
        <Shield className="h-6 w-6 mr-2" />
        <span className="text-lg font-bold">NeuraShield</span>
      </div>
      <div className="flex flex-col py-2">
        {menuItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`flex items-center px-4 py-3 text-sm ${
              isActive(item.href)
                ? "bg-[#1a1d29] text-white font-medium"
                : "text-gray-400 hover:text-white hover:bg-[#1a1d29]"
            }`}
          >
            <item.icon className="mr-3 h-5 w-5" />
            {item.title}
          </Link>
        ))}
      </div>
      <div className="absolute bottom-0 w-full border-t border-gray-800">
        <Link
          href="/dashboard/notifications"
          className="flex items-center px-4 py-3 text-sm text-gray-400 hover:text-white hover:bg-[#1a1d29]"
        >
          <Bell className="mr-3 h-5 w-5" />
          Notifications
        </Link>
        <button
          onClick={logout}
          className="flex w-full items-center px-4 py-3 text-sm text-gray-400 hover:text-white hover:bg-[#1a1d29]"
        >
          <LogOut className="mr-3 h-5 w-5" />
          Logout
        </button>
      </div>
    </div>
  )
}
