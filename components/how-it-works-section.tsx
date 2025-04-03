"use client"

import { useState } from "react"
import { Search, Database, FileText, FileCheck, ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"

export default function HowItWorksSection() {
  const [activeTab, setActiveTab] = useState(0)

  // Step data
  const steps = [
    {
      id: "detect",
      icon: Search,
      title: "AI Detects Threats",
      description: "Scans systems in real-time with machine learning precision",
      detail: "99.9% threat detection rate with continuous learning algorithms",
      metric: "0.027 seconds",
      metricLabel: "Average scan time",
      color: "primary",
      apiEndpoint: "/api/steps/detect",
    },
    {
      id: "log",
      icon: Database,
      title: "Logs on Hyperledger",
      description: "Records every event immutably on a blockchain",
      detail: "Tamper-proof evidence secured across distributed nodes",
      metric: "2.4 seconds",
      metricLabel: "Blockchain confirmation",
      color: "secondary",
      apiEndpoint: "/api/steps/log",
    },
    {
      id: "analyze",
      icon: FileText,
      title: "Analyzes Evidence",
      description: "Processes data into actionable insights",
      detail: "Forensic-grade analysis correlates events across systems",
      metric: "99.8%",
      metricLabel: "Analysis accuracy",
      color: "primary",
      apiEndpoint: "/api/steps/analyze",
    },
    {
      id: "deliver",
      icon: FileCheck,
      title: "Delivers Proof",
      description: "Generates court-admissible reports",
      detail: "Legal-grade documentation accepted in all 50 states",
      metric: "100%",
      metricLabel: "Court acceptance rate",
      color: "secondary",
      apiEndpoint: "/api/steps/deliver",
    },
  ]

  // Handle tab change
  const handleTabChange = (index: number) => {
    setActiveTab(index)
    console.log(`Step clicked: ${steps[index].id}`)
    console.log(`API call would be made to: ${steps[index].apiEndpoint}`)
  }

  return (
    <section className="section-container bg-muted/20" id="how-it-works">
      <h2 className="section-title fade-in">
        How <span className="gradient-text">NeuraShield</span> Works
      </h2>
      <p className="section-subtitle fade-in">
        Our four-step process ensures complete protection and legal defensibility
      </p>

      {/* Process visualization */}
      <div className="max-w-5xl mx-auto mt-12 px-4">
        <div className="flex flex-col lg:flex-row gap-8 lg:gap-12">
          {/* Step navigation */}
          <div className="lg:w-1/3">
            <div className="bg-card/50 rounded-lg border border-white/10 p-4">
              <div className="space-y-2">
                {steps.map((step, index) => (
                  <button
                    key={step.id}
                    onClick={() => handleTabChange(index)}
                    className={cn(
                      "w-full text-left p-4 rounded-lg transition-all duration-300 flex items-center gap-4",
                      activeTab === index
                        ? "bg-gradient-to-r from-primary/20 to-secondary/20 border border-white/20"
                        : "hover:bg-white/5",
                    )}
                    data-step-id={step.id}
                    data-api-endpoint={step.apiEndpoint}
                  >
                    <div
                      className={cn(
                        "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center",
                        activeTab === index
                          ? "bg-gradient-to-r from-primary to-secondary text-white"
                          : "bg-muted/50 text-muted-foreground",
                      )}
                    >
                      {index + 1}
                    </div>
                    <div>
                      <h3 className={cn("font-medium", activeTab === index ? "text-white" : "text-muted-foreground")}>
                        {step.title}
                      </h3>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Step content */}
          <div className="lg:w-2/3">
            <div className="bg-card/50 rounded-lg border border-white/10 p-6 h-full">
              {steps.map((step, index) => (
                <div
                  key={step.id}
                  className={cn(
                    "transition-opacity duration-300",
                    activeTab === index ? "block opacity-100" : "hidden opacity-0",
                  )}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className={`p-3 rounded-full bg-${step.color}/20`}>
                      <step.icon className={`h-6 w-6 text-${step.color}`} />
                    </div>
                    <h3 className="text-2xl font-bold">{step.title}</h3>
                  </div>

                  <p className="text-lg text-muted-foreground mb-6">{step.description}</p>

                  <div className="bg-muted/30 rounded-lg p-4 mb-6 border border-white/10">
                    <p className="mb-4">{step.detail}</p>
                    <div className="flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{step.metricLabel}:</span>
                      <span className={`text-sm font-medium text-${step.color}`}>{step.metric}</span>
                    </div>
                  </div>

                  {/* Step illustration */}
                  <div className="mt-8 h-48 rounded-lg bg-gradient-to-br from-black to-muted/50 flex items-center justify-center border border-white/5">
                    <div className="text-center">
                      <step.icon className={`h-12 w-12 text-${step.color} mx-auto mb-4`} />
                      <div className="text-sm text-muted-foreground">Step {index + 1} visualization</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Process time indicator */}
      <div className="mt-16 text-center fade-in">
        <div className="inline-block max-w-md mx-auto bg-card/50 border border-white/10 rounded-lg p-6">
          <div className="text-sm text-muted-foreground mb-2">Average time from detection to court-ready report</div>
          <div className="text-3xl font-bold gradient-text" data-api-endpoint="/metrics/total-process-time">
            4.7 minutes
          </div>
        </div>
      </div>

      {/* Process flow diagram */}
      <div className="mt-16 max-w-4xl mx-auto px-4 fade-in">
        <div className="bg-card/30 rounded-lg border border-white/10 p-6">
          <h3 className="text-xl font-bold mb-6 text-center">The NeuraShield Process Flow</h3>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {steps.map((step, index) => (
              <div key={step.id} className="text-center">
                <div className="relative">
                  <div
                    className={`w-16 h-16 rounded-full bg-${step.color}/20 flex items-center justify-center mx-auto`}
                  >
                    <step.icon className={`h-8 w-8 text-${step.color}`} />
                  </div>

                  {/* Connector line */}
                  {index < steps.length - 1 && (
                    <div className="hidden md:block absolute top-1/2 left-full w-full h-0.5 bg-gradient-to-r from-primary to-secondary transform -translate-y-1/2">
                      <div className="absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-secondary"></div>
                    </div>
                  )}
                </div>

                <h4 className="mt-3 font-medium text-sm">{step.title}</h4>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

