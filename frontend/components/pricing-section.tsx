"use client"

import { useState } from "react"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Check } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { useSignupModal } from "@/hooks/use-signup-modal"

// Pricing data
const pricingPlans = {
  monthly: [
    {
      id: "basic",
      name: "Basic",
      price: "$29",
      period: "mo",
      features: [
        "Core AI protection",
        "500GB Hyperledger storage",
        "Basic threat reports",
        "Email alerts",
        "8/5 support",
      ],
    },
    {
      id: "pro",
      name: "Pro",
      price: "$49",
      period: "mo",
      popular: true,
      features: [
        "Advanced AI protection",
        "2TB Hyperledger storage",
        "Detailed forensic analysis",
        "Real-time alerts (email, SMS)",
        "24/7 priority support",
        "Court-ready report generation",
      ],
    },
    {
      id: "enterprise",
      name: "Enterprise",
      price: "$99",
      period: "mo",
      features: [
        "Full AI security suite",
        "Unlimited Hyperledger storage",
        "Advanced threat hunting",
        "Custom integration",
        "Dedicated security team",
        "Legal support package",
      ],
    },
  ],
  yearly: [
    {
      id: "basic-yearly",
      name: "Basic",
      price: "$290",
      period: "yr",
      features: [
        "Core AI protection",
        "500GB Hyperledger storage",
        "Basic threat reports",
        "Email alerts",
        "8/5 support",
      ],
    },
    {
      id: "pro-yearly",
      name: "Pro",
      price: "$490",
      period: "yr",
      popular: true,
      features: [
        "Advanced AI protection",
        "2TB Hyperledger storage",
        "Detailed forensic analysis",
        "Real-time alerts (email, SMS)",
        "24/7 priority support",
        "Court-ready report generation",
      ],
    },
    {
      id: "enterprise-yearly",
      name: "Enterprise",
      price: "$990",
      period: "yr",
      features: [
        "Full AI security suite",
        "Unlimited Hyperledger storage",
        "Advanced threat hunting",
        "Custom integration",
        "Dedicated security team",
        "Legal support package",
      ],
    },
  ],
}

export default function PricingSection() {
  const [billingPeriod, setBillingPeriod] = useState<"monthly" | "yearly">("monthly")
  const { openModal } = useSignupModal()

  // Toggle billing period
  const toggleBillingPeriod = () => {
    setBillingPeriod((prev) => (prev === "monthly" ? "yearly" : "monthly"))
  }

  // Get current plans based on billing period
  const currentPlans = pricingPlans[billingPeriod]

  return (
    <section className="section-container" id="pricing">
      <h2 className="section-title fade-in">
        Simple, Transparent <span className="gradient-text">Pricing</span>
      </h2>
      <p className="section-subtitle fade-in">Choose the plan that's right for your organization's security needs</p>

      <div className="mt-8 text-center fade-in" data-pricing-endpoint="/api/pricing">
        <div className="inline-flex items-center mb-12">
          <span className={`mr-2 ${billingPeriod === "monthly" ? "text-white" : "text-muted-foreground"}`}>
            Monthly
          </span>
          <Switch
            checked={billingPeriod === "yearly"}
            onCheckedChange={toggleBillingPeriod}
            className="data-[state=checked]:bg-primary"
          />
          <span className={`ml-2 ${billingPeriod === "yearly" ? "text-white" : "text-muted-foreground"}`}>
            Yearly{" "}
            <span className="ml-1 text-xs bg-secondary/20 text-secondary px-2 py-0.5 rounded-full">Save 17%</span>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {currentPlans.map((plan, index) => (
            <Card
              key={plan.id}
              className={`bg-card/50 border ${
                plan.popular ? "border-primary/50 scale-105 md:scale-110 z-10" : "border-white/10"
              } overflow-hidden card-glow fade-in`}
              style={{ transitionDelay: `${0.2 * index}s` }}
              data-plan-id={plan.id}
            >
              {plan.popular && (
                <div className="absolute top-0 right-0 bg-primary text-primary-foreground text-xs font-bold px-3 py-1">
                  MOST POPULAR
                </div>
              )}
              <div
                className={`h-2 ${
                  plan.popular
                    ? "bg-gradient-to-r from-primary to-secondary"
                    : index === 0
                      ? "bg-primary/50"
                      : "bg-secondary/50"
                }`}
              ></div>
              <CardHeader className="pt-6">
                <h3 className="text-xl font-bold">{plan.name}</h3>
                <div className="mt-4">
                  <span className="text-4xl font-bold">{plan.price}</span>
                  <span className="text-muted-foreground">/{plan.period}</span>
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  {plan.name === "Basic"
                    ? "Essential protection for small businesses"
                    : plan.name === "Pro"
                      ? "Advanced protection for growing organizations"
                      : "Comprehensive protection for large enterprises"}
                </p>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-3">
                  {plan.features.map((feature, i) => (
                    <li key={i} className="flex items-start">
                      <Check
                        className={`w-5 h-5 mr-2 flex-shrink-0 mt-0.5 ${
                          plan.popular ? "text-primary" : index === 0 ? "text-primary" : "text-secondary"
                        }`}
                      />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button
                  className={`w-full ${
                    plan.popular ? "bg-gradient-to-r from-primary to-secondary hover:opacity-90" : ""
                  }`}
                  variant={plan.popular ? "default" : "outline"}
                  onClick={() => openModal(plan.id)}
                  data-signup-endpoint={`/api/signup/${plan.id}`}
                >
                  Sign Up
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>

      <div className="mt-16 text-center max-w-2xl mx-auto fade-in" style={{ transitionDelay: "0.6s" }}>
        <Card className="bg-muted/30 border border-white/10">
          <CardContent className="p-6">
            <h3 className="text-lg font-bold mb-2">Need a custom solution?</h3>
            <p className="text-muted-foreground mb-4">
              Our security experts can create a tailored plan for your organization's specific needs.
            </p>
            <Button
              variant="outline"
              className="bg-transparent"
              onClick={() => openModal("custom")}
              data-contact-endpoint="/api/contact/custom"
            >
              Contact Our Team
            </Button>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}

