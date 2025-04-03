"use client"
import { cn } from "@/lib/utils"
import { Shield, Bell, User } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export default function Header() {
  const pathname = usePathname()

  const links = [
    { name: "Dashboard", href: "/" },
    { name: "Logs", href: "/logs" },
    { name: "Settings", href: "/settings" },
  ]

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/10 glass-effect">
      <div className="container flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="h-8 w-8 text-primary glow" />
            <span className="text-xl font-bold glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              NeuraShield
            </span>
          </Link>

          <nav className="hidden md:flex gap-6">
            {links.map((link) => (
              <Link
                key={link.name}
                href={link.href}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-primary",
                  pathname === link.href ? "text-primary border-b-2 border-primary" : "text-muted-foreground",
                )}
              >
                {link.name}
              </Link>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon">
            <Bell className="h-5 w-5" />
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center">
                  <User className="h-4 w-4" />
                </div>
                <span className="hidden md:inline-block">Admin</span>
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
    </header>
  )
}

