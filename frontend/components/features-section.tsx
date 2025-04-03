"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, Database, Shield, ChevronRight } from "lucide-react"

export default function FeaturesSection() {
  // Handle smooth scrolling to How It Works section
  const scrollToHowItWorks = () => {
    const section = document.getElementById("how-it-works")
    if (section) {
      section.scrollIntoView({ behavior: "smooth" })
    }
  }

  return (
    <section className="section-container" id="features">
      <h2 className="section-title fade-in">
        Cutting-Edge <span className="gradient-text">Features</span>
      </h2>
      <p className="section-subtitle fade-in">
        NeuraShield combines advanced AI with Hyperledger Fabric to provide unmatched security and legal protection
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
        {/* Feature 1 */}
        <Card
          className="bg-card/50 border border-white/10 overflow-hidden card-glow fade-in feature-card"
          data-feature-id="ai-detection"
        >
          <div className="h-2 bg-gradient-to-r from-primary to-primary/50"></div>
          <CardContent className="pt-6">
            <div className="w-14 h-14 rounded-full bg-primary/20 flex items-center justify-center mb-6 icon-placeholder">
              <Brain className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-bold mb-3">AI Threat Detection</h3>
            <p className="text-muted-foreground mb-4">
              Real-time monitoring with AI precision that identifies and neutralizes threats before they cause damage.
            </p>
            <div className="mt-6 pt-4 border-t border-white/10">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Threats detected today</span>
                <span className="text-lg font-bold text-primary" data-api-endpoint="/stats/threats-today">
                  42
                </span>
              </div>
              <Button variant="ghost" className="text-primary mt-4 p-0 h-auto learn-more" onClick={scrollToHowItWorks}>
                Learn More <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Feature 2 */}
        <Card
          className="bg-card/50 border border-white/10 overflow-hidden card-glow fade-in feature-card"
          style={{ transitionDelay: "0.2s" }}
          data-feature-id="hyperledger-proof"
        >
          <div className="h-2 bg-gradient-to-r from-secondary to-secondary/50"></div>
          <CardContent className="pt-6">
            <div className="w-14 h-14 rounded-full bg-secondary/20 flex items-center justify-center mb-6 icon-placeholder">
              <Database className="w-8 h-8 text-secondary" />
            </div>
            <h3 className="text-xl font-bold mb-3">Hyperledger Proof</h3>
            <p className="text-muted-foreground mb-4">
              Immutable logs for legal use stored on Hyperledger Fabric blockchain provide tamper-proof evidence.
            </p>
            <div className="mt-6 pt-4 border-t border-white/10">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Records secured</span>
                <span className="text-lg font-bold text-secondary" data-api-endpoint="/stats/records-secured">
                  10K+
                </span>
              </div>
              <Button
                variant="ghost"
                className="text-secondary mt-4 p-0 h-auto learn-more"
                onClick={scrollToHowItWorks}
              >
                Learn More <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Feature 3 */}
        <Card
          className="bg-card/50 border border-white/10 overflow-hidden card-glow fade-in feature-card"
          style={{ transitionDelay: "0.4s" }}
          data-feature-id="end-to-end"
        >
          <div className="h-2 bg-gradient-to-r from-primary to-secondary"></div>
          <CardContent className="pt-6">
            <div className="w-14 h-14 rounded-full bg-gradient-to-r from-primary/20 to-secondary/20 flex items-center justify-center mb-6 icon-placeholder">
              <Shield className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-bold mb-3">End-to-End Security</h3>
            <p className="text-muted-foreground mb-4">
              Comprehensive system protection that safeguards your data, networks, and systems from every attack vector.
            </p>
            <div className="mt-6 pt-4 border-t border-white/10">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Protection coverage</span>
                <span className="text-lg font-bold gradient-text" data-api-endpoint="/stats/protection-coverage">
                  100%
                </span>
              </div>
              <Button variant="ghost" className="text-primary mt-4 p-0 h-auto learn-more" onClick={scrollToHowItWorks}>
                Learn More <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}

