import type React from "react"
import { DashboardSidebar } from "@/components/dashboard/dashboard-sidebar"
import { DashboardHeader } from "@/components/dashboard/dashboard-header"
import type { Metadata } from "next"

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
    <div className="flex min-h-screen bg-[#0a0c10] text-white">
      <DashboardSidebar />
      <div className="flex flex-1 flex-col">
        <DashboardHeader />
        <main className="flex-1 p-6 pt-16">{children}</main>
      </div>
    </div>
  )
}
