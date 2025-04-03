"use client"

import { useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Shield, Lock, Database } from "lucide-react"
import { useSignupModal } from "@/hooks/use-signup-modal"

export default function HeroSection() {
  const { openModal } = useSignupModal()
  const containerRef = useRef<HTMLDivElement>(null)

  // Create blockchain animation nodes
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const blockchainBg = document.createElement("div")
    blockchainBg.className = "blockchain-bg"
    container.appendChild(blockchainBg)

    // Create nodes
    const nodeCount = 15
    for (let i = 0; i < nodeCount; i++) {
      const node = document.createElement("div")
      node.className = "node"
      node.style.left = `${Math.random() * 100}%`
      node.style.top = `${Math.random() * 100}%`
      node.style.animationDelay = `${Math.random() * 5}s`

      // Different colors for some nodes
      if (i % 3 === 0) {
        node.style.background = "var(--accent-purple)"
      }

      blockchainBg.appendChild(node)
    }

    return () => {
      if (container.contains(blockchainBg)) {
        container.removeChild(blockchainBg)
      }
    }
  }, [])

  return (
    <section
      ref={containerRef}
      className="relative min-h-[calc(100vh-4rem)] flex items-center overflow-hidden"
      id="hero"
      data-video-src="/videos/blockchain-animation.mp4"
    >
      {/* Content */}
      <div className="container relative z-10 py-20">
        <div className="max-w-3xl mx-auto text-center">
          <div className="fade-in">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
              <span className="gradient-text">NeuraShield:</span> AI Security, Court-Ready Proof
            </h1>
          </div>

          <div className="fade-in" style={{ transitionDelay: "0.2s" }}>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8">
              Protect your systems with AI and Hyperledger-backed evidence
            </p>
          </div>

          <div className="fade-in" style={{ transitionDelay: "0.4s" }}>
            <Button
              size="lg"
              className="bg-gradient-to-r from-primary to-secondary text-white px-8 py-6 text-lg rounded-full hover:scale-105 transition-transform pulse-animation"
              onClick={() => openModal()}
              data-signup-endpoint="/api/signup"
            >
              Start Free Trial
            </Button>
          </div>

          <div className="mt-12 grid grid-cols-3 gap-4 max-w-lg mx-auto fade-in" style={{ transitionDelay: "0.6s" }}>
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-2">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground">Advanced Protection</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-secondary/20 flex items-center justify-center mb-2">
                <Lock className="w-6 h-6 text-secondary" />
              </div>
              <p className="text-sm text-muted-foreground">Court Admissible</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-2">
                <Database className="w-6 h-6 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground">Hyperledger Backed</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

