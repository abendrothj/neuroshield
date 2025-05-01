import type React from "react"
import { DashboardSidebar } from "@/components/dashboard/dashboard-sidebar"
import { DashboardHeader } from "@/components/dashboard/dashboard-header"
import type { Metadata } from "next"
import { SidebarNav } from "@/components/dashboard/sidebar-nav"

export const metadata: Metadata = {
  title: "Dashboard - NeuraShield",
  description: "NeuraShield dashboard for threat monitoring and management",
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen">
      <SidebarNav />
      <main>{children}</main>
    </div>
  )
}
