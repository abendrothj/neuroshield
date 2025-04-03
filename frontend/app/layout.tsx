import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { SignupModalProvider } from "@/hooks/use-signup-modal"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "NeuraShield - AI-Powered Cybersecurity with Court-Ready Evidence",
  description:
    "NeuraShield provides unbreakable security with AI protection and Hyperledger Fabric-backed proof for court-ready evidence.",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body className={`${inter.className} bg-background text-foreground`}>
        <SignupModalProvider>{children}</SignupModalProvider>
      </body>
    </html>
  )
}



import './globals.css'