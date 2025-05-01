import { usePathname } from "next/navigation"
import Link from "next/link"
import { BarChart3, Shield, Database, Brain, HardDrive, Settings, Users, Bell, LogOut, FileText } from "lucide-react"
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
      title: "Evidence",
      icon: FileText,
      href: "/dashboard/evidence",
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
    <div className="flex flex-col h-full">
      {/* Rest of the component code remains unchanged */}
    </div>
  )
} 