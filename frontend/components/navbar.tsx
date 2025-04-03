"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Shield, Menu, X } from "lucide-react"
import { useSignupModal } from "@/hooks/use-signup-modal"

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const { openModal } = useSignupModal()

  // Handle scroll event to change navbar appearance
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Handle smooth scrolling for navigation links
  const scrollToSection = (sectionId: string) => {
    setIsMenuOpen(false)
    const section = document.getElementById(sectionId)
    if (section) {
      section.scrollIntoView({ behavior: "smooth" })
    }
  }

  return (
    <header
      className={`sticky top-0 z-50 w-full border-b transition-all duration-300 ${
        isScrolled ? "border-white/10 bg-background/80 backdrop-blur-md" : "border-transparent bg-transparent"
      }`}
    >
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="h-8 w-8 text-primary" />
          <Link href="/" className="text-xl font-bold gradient-text">
            NeuraShield
          </Link>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-6">
          <button
            onClick={() => scrollToSection("features")}
            className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
          >
            Features
          </button>
          <button
            onClick={() => scrollToSection("how-it-works")}
            className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
          >
            How It Works
          </button>
          <button
            onClick={() => scrollToSection("evidence")}
            className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
          >
            Evidence
          </button>
          <button
            onClick={() => scrollToSection("testimonials")}
            className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
          >
            Testimonials
          </button>
          <button
            onClick={() => scrollToSection("pricing")}
            className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
          >
            Pricing
          </button>
          <Button variant="outline" className="ml-4" onClick={() => openModal()} data-login-endpoint="/api/login">
            Log In
          </Button>
          <Button
            className="bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-all hover:scale-105"
            onClick={() => openModal()}
            data-signup-endpoint="/api/signup"
          >
            Sign Up
          </Button>
        </nav>

        {/* Mobile Menu Button */}
        <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsMenuOpen(!isMenuOpen)}>
          {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </Button>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden border-b border-white/10 bg-background">
          <nav className="flex flex-col p-4 space-y-4">
            <button
              onClick={() => scrollToSection("features")}
              className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors p-2"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("how-it-works")}
              className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors p-2"
            >
              How It Works
            </button>
            <button
              onClick={() => scrollToSection("evidence")}
              className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors p-2"
            >
              Evidence
            </button>
            <button
              onClick={() => scrollToSection("testimonials")}
              className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors p-2"
            >
              Testimonials
            </button>
            <button
              onClick={() => scrollToSection("pricing")}
              className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors p-2"
            >
              Pricing
            </button>
            <div className="flex flex-col space-y-2 pt-2">
              <Button
                variant="outline"
                onClick={() => {
                  setIsMenuOpen(false)
                  openModal()
                }}
                data-login-endpoint="/api/login"
              >
                Log In
              </Button>
              <Button
                className="bg-gradient-to-r from-primary to-secondary hover:opacity-90"
                onClick={() => {
                  setIsMenuOpen(false)
                  openModal()
                }}
                data-signup-endpoint="/api/signup"
              >
                Sign Up
              </Button>
            </div>
          </nav>
        </div>
      )}
    </header>
  )
}

