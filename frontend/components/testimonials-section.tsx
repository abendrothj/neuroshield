"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Quote } from "lucide-react"

// Testimonial data - would be fetched from API in real implementation
const testimonials = [
  {
    id: 1,
    quote:
      "NeuraShield saved us in court! The blockchain-backed evidence was irrefutable and helped us win our case against the attackers.",
    name: "Jane Doe",
    title: "CTO",
    company: "SecureCorp",
  },
  {
    id: 2,
    quote:
      "As a financial institution, we're under constant attack. NeuraShield not only protected us but provided us with legally defensible evidence.",
    name: "Michael Chen",
    title: "CISO",
    company: "Global Financial",
  },
  {
    id: 3,
    quote:
      "The combination of AI detection and Hyperledger proof is revolutionary. We've reduced our security incidents by 94% since implementing NeuraShield.",
    name: "Jessica Williams",
    title: "VP of Security",
    company: "HealthTech Solutions",
  },
]

export default function TestimonialsSection() {
  const [activeIndex, setActiveIndex] = useState(0)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  // Handle navigation
  const nextTestimonial = () => {
    setActiveIndex((prev) => (prev + 1) % testimonials.length)
  }

  const prevTestimonial = () => {
    setActiveIndex((prev) => (prev - 1 + testimonials.length) % testimonials.length)
  }

  const goToTestimonial = (index: number) => {
    setActiveIndex(index)

    // Reset the interval when manually navigating
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      startAutoRotation()
    }
  }

  // Auto-rotation functionality
  const startAutoRotation = () => {
    intervalRef.current = setInterval(() => {
      nextTestimonial()
    }, 5000)
  }

  useEffect(() => {
    startAutoRotation()

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  return (
    <section className="section-container bg-muted/20" id="testimonials">
      <h2 className="section-title fade-in">
        What Our <span className="gradient-text">Clients</span> Say
      </h2>
      <p className="section-subtitle fade-in">
        Trusted by leading organizations to provide unbreakable security and court-ready evidence
      </p>

      <div className="mt-12 max-w-3xl mx-auto relative fade-in">
        <div className="relative h-[300px]" data-testimonial-endpoint="/api/testimonials">
          {testimonials.map((testimonial, index) => (
            <div key={testimonial.id} className={`testimonial-slide ${index === activeIndex ? "active" : ""}`}>
              <Card className="bg-card/50 border border-white/10 h-full">
                <CardContent className="p-8 h-full flex flex-col">
                  <Quote className="w-12 h-12 text-primary/30 mb-6" />
                  <p className="text-lg mb-8 italic flex-grow">"{testimonial.quote}"</p>
                  <div className="flex items-center">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-r from-primary/20 to-secondary/20 flex items-center justify-center mr-4">
                      <span className="text-lg font-bold gradient-text">{testimonial.name.charAt(0)}</span>
                    </div>
                    <div>
                      <div className="font-bold">{testimonial.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {testimonial.title}, {testimonial.company}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>

        {/* Navigation */}
        <div className="flex justify-center mt-8 items-center">
          <Button variant="outline" size="icon" className="mr-4 rounded-full" onClick={prevTestimonial}>
            <ChevronLeft className="w-5 h-5" />
          </Button>

          {testimonials.map((_, index) => (
            <button
              key={index}
              className={`w-2.5 h-2.5 rounded-full mx-1 transition-colors duration-300 ${
                index === activeIndex ? "bg-primary" : "bg-muted-foreground/30"
              }`}
              onClick={() => goToTestimonial(index)}
            />
          ))}

          <Button variant="outline" size="icon" className="ml-4 rounded-full" onClick={nextTestimonial}>
            <ChevronRight className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </section>
  )
}

