"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  BarChart3Icon,
  LockIcon,
  ServerIcon,
  ShieldAlertIcon,
  ShieldCheckIcon,
  UsersIcon,
  LayersIcon,
  ActivityIcon,
  BrainCircuitIcon,
  DatabaseIcon,
} from "lucide-react";

interface SidebarNavProps extends React.HTMLAttributes<HTMLElement> {
  items?: {
    href: string;
    title: string;
    icon: React.ReactNode;
  }[];
}

export function SidebarNav({ className, items, ...props }: SidebarNavProps) {
  const pathname = usePathname();

  // Default navigation items if none provided
  const defaultItems = [
    {
      href: "/dashboard",
      title: "Overview",
      icon: <BarChart3Icon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/ai-models",
      title: "AI Models",
      icon: <BrainCircuitIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/blockchain",
      title: "Blockchain",
      icon: <LayersIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/threats",
      title: "Threats",
      icon: <ShieldAlertIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/security",
      title: "Security",
      icon: <ShieldCheckIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/users",
      title: "Users",
      icon: <UsersIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/ipfs",
      title: "IPFS Storage",
      icon: <DatabaseIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/api",
      title: "API",
      icon: <ActivityIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/servers",
      title: "Servers",
      icon: <ServerIcon className="w-5 h-5" />,
    },
    {
      href: "/dashboard/auth",
      title: "Authentication",
      icon: <LockIcon className="w-5 h-5" />,
    },
  ];

  const navItems = items || defaultItems;

  return (
    <nav
      className={cn(
        "flex flex-col space-y-1 h-screen bg-background fixed top-0 left-0 w-64 border-r p-4 pt-8",
        className
      )}
      {...props}
    >
      <div className="flex items-center mb-6 px-2">
        <ShieldCheckIcon className="w-8 h-8 mr-2 text-primary" />
        <h2 className="text-xl font-bold">NeuraShield</h2>
      </div>
      
      {navItems.map((item) => {
        const isActive = pathname === item.href;
        return (
          <Button
            key={item.href}
            variant={isActive ? "secondary" : "ghost"}
            className={cn(
              "w-full justify-start text-sm font-medium",
              isActive ? "bg-secondary text-secondary-foreground" : "text-muted-foreground hover:text-foreground"
            )}
            asChild
          >
            <Link href={item.href}>
              <div className="flex items-center">
                {item.icon}
                <span className="ml-3">{item.title}</span>
              </div>
            </Link>
          </Button>
        );
      })}
    </nav>
  );
} 