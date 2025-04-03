"use client"

import { useEffect } from "react"
import Navbar from "@/components/navbar"
import HeroSection from "@/components/hero-section"
import FeaturesSection from "@/components/features-section"
import HowItWorksSection from "@/components/how-it-works-section"
import EvidenceShowcase from "@/components/evidence-showcase"
import TestimonialsSection from "@/components/testimonials-section"
import PricingSection from "@/components/pricing-section"
import Footer from "@/components/footer"
import SignupModal from "@/components/signup-modal"
import { useSignupModal } from "@/hooks/use-signup-modal"

export default function Home() {
  const { isOpen, selectedPlan } = useSignupModal()

  // Fade-in effect for elements on scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible")
          }
        })
      },
      { threshold: 0.1 },
    )

    const fadeElements = document.querySelectorAll(".fade-in")
    fadeElements.forEach((el) => observer.observe(el))

    return () => {
      fadeElements.forEach((el) => observer.unobserve(el))
    }
  }, [])

  return (
    <main className="flex min-h-screen flex-col">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <HowItWorksSection />
      <EvidenceShowcase />
      <TestimonialsSection />
      <PricingSection />
      <Footer />
      {isOpen && <SignupModal plan={selectedPlan} />}
    </main>
  )
}

