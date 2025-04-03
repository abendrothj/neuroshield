"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

type SignupModalContextType = {
  isOpen: boolean
  selectedPlan: string | null
  openModal: (plan?: string) => void
  closeModal: () => void
}

const SignupModalContext = createContext<SignupModalContextType | undefined>(undefined)

export function SignupModalProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null)

  const openModal = (plan?: string) => {
    setSelectedPlan(plan || null)
    setIsOpen(true)
    document.body.style.overflow = "hidden" // Prevent scrolling when modal is open
  }

  const closeModal = () => {
    setIsOpen(false)
    document.body.style.overflow = "" // Restore scrolling
  }

  return (
    <SignupModalContext.Provider value={{ isOpen, selectedPlan, openModal, closeModal }}>
      {children}
    </SignupModalContext.Provider>
  )
}

export function useSignupModal() {
  const context = useContext(SignupModalContext)
  if (context === undefined) {
    throw new Error("useSignupModal must be used within a SignupModalProvider")
  }
  return context
}

