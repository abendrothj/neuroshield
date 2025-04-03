"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { X } from "lucide-react"
import { useSignupModal } from "@/hooks/use-signup-modal"

interface SignupModalProps {
  plan: string | null
}

export default function SignupModal({ plan }: SignupModalProps) {
  const { closeModal } = useSignupModal()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    // Validate email
    const emailRegex = /^[^@]+@[^@]+\.[^@]+$/
    if (!emailRegex.test(email)) {
      alert("Please enter a valid email address")
      return
    }

    // Validate password
    if (password.length < 8) {
      alert("Password must be at least 8 characters long")
      return
    }

    setIsSubmitting(true)

    // Simulate API call
    setTimeout(() => {
      // Log form data for backend integration
      console.log("Signup form submitted:", {
        email,
        password: "********", // Don't log actual password
        plan,
      })
      console.log("API call would be made to:", "/api/signup", {
        email,
        password: "********",
        plan,
      })

      setIsSubmitting(false)
      setSuccess(true)

      // Close modal after showing success message
      setTimeout(() => {
        closeModal()
      }, 2000)
    }, 1000)
  }

  // Handle modal close
  const handleClose = () => {
    if (!isSubmitting) {
      closeModal()
    }
  }

  // Handle click outside to close
  const handleOutsideClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && !isSubmitting) {
      closeModal()
    }
  }

  return (
    <div className="modal-overlay" onClick={handleOutsideClick}>
      <div className="modal-content">
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2"
          onClick={handleClose}
          disabled={isSubmitting}
        >
          <X className="h-4 w-4" />
        </Button>

        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold">
            {success
              ? "Trial Started!"
              : plan
                ? `Sign Up for ${plan.charAt(0).toUpperCase() + plan.slice(1).replace(/-yearly$/, " Yearly")} Plan`
                : "Start Your Free Trial"}
          </h2>
          <p className="text-muted-foreground mt-2">
            {success ? "Check your email for next steps" : "Get started with NeuraShield in just a few seconds"}
          </p>
        </div>

        {!success ? (
          <form onSubmit={handleSubmit} data-signup-endpoint="/api/signup">
            {plan && <input type="hidden" name="plan" value={plan} />}

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@company.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={isSubmitting}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={isSubmitting}
                />
              </div>

              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-primary to-secondary"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Processing..." : "Submit"}
              </Button>

              <p className="text-xs text-center text-muted-foreground mt-4">
                By signing up, you agree to our Terms of Service and Privacy Policy
              </p>
            </div>
          </form>
        ) : (
          <div className="text-center py-4">
            <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-lg">Your trial has been activated!</p>
            <p className="text-muted-foreground mt-2">Check your email for instructions to get started.</p>
          </div>
        )}
      </div>
    </div>
  )
}

