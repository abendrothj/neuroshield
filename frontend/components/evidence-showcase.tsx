"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FileText, Download, RefreshCw } from "lucide-react"

// Sample report data
const initialReport = {
  date: "04/01/2025",
  threat: "Malware",
  hash: "0x7d9f1c8a2e3b5d4f6g7h8j9k0l1m2n3o4p",
  status: "Logged",
}

export default function EvidenceShowcase() {
  const [report, setReport] = useState(initialReport)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Handle download sample report
  const handleDownload = () => {
    console.log("Downloading sample report:", report)
    alert("Sample report downloaded")

    // Log action for backend integration
    console.log("API call would be made to:", "/api/reports/download", {
      reportId: "sample1",
      reportData: report,
    })
  }

  // Handle refresh report
  const handleRefresh = () => {
    setIsRefreshing(true)

    // Simulate API call
    setTimeout(() => {
      // Generate new random report data
      const threats = ["Malware", "Phishing", "DDoS", "Ransomware", "Data Breach"]
      const newReport = {
        date: new Date().toLocaleDateString(),
        threat: threats[Math.floor(Math.random() * threats.length)],
        hash: "0x" + Math.random().toString(16).substring(2, 30),
        status: Math.random() > 0.2 ? "Logged" : "Blocked",
      }

      setReport(newReport)
      setIsRefreshing(false)

      // Log action for backend integration
      console.log("API call would be made to:", "/api/reports/refresh")
    }, 1000)
  }

  return (
    <section className="section-container" id="evidence">
      <h2 className="section-title fade-in">
        Court-Ready <span className="gradient-text">Evidence</span>
      </h2>
      <p className="section-subtitle fade-in">
        NeuraShield provides tamper-proof, legally admissible evidence backed by Hyperledger Fabric
      </p>

      <div className="mt-12 max-w-2xl mx-auto">
        <Card
          className="bg-card/50 border border-white/10 overflow-hidden shadow-lg card-glow fade-in"
          data-report-id="sample1"
        >
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <FileText className="w-6 h-6 text-primary mr-2" />
                <h3 className="text-xl font-bold">Threat Report</h3>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="text-muted-foreground hover:text-primary"
                onClick={handleRefresh}
                disabled={isRefreshing}
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>

            <div className="bg-muted/30 rounded-lg p-6 mb-6 border border-white/10">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Date</div>
                  <div className="font-mono text-sm">{report.date}</div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground mb-1">Threat</div>
                  <div className="font-mono text-sm">{report.threat}</div>
                </div>

                <div className="col-span-2">
                  <div className="text-sm text-muted-foreground mb-1">Hyperledger Hash</div>
                  <div className="font-mono text-xs break-all">{report.hash}</div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground mb-1">Status</div>
                  <div
                    className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                      report.status === "Blocked" ? "bg-red-900/30 text-red-400" : "bg-green-900/30 text-green-400"
                    }`}
                  >
                    {report.status}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-muted/30 p-4 rounded-md mb-6">
              <p className="text-sm mb-2">
                This report has been cryptographically verified on Hyperledger Fabric and is legally admissible in court
                proceedings.
              </p>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                <span className="text-xs text-green-400">Blockchain verification: VALID</span>
              </div>
            </div>

            <Button
              className="w-full bg-gradient-to-r from-primary to-secondary"
              onClick={handleDownload}
              data-download-endpoint="/api/reports/download"
            >
              <Download className="w-4 h-4 mr-2" />
              Download Sample
            </Button>
          </div>
        </Card>
      </div>
    </section>
  )
}

