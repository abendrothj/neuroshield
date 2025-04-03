import { Database, Shield } from "lucide-react"

interface FooterProps {
  lastAction?: string
}

export default function Footer({ lastAction = "None" }: FooterProps) {
  return (
    <footer className="border-t border-white/10 bg-background/80 backdrop-blur-sm py-3">
      <div className="container flex flex-col sm:flex-row justify-between items-center gap-2">
        <p className="text-sm text-muted-foreground flex items-center">
          <Shield className="mr-2 h-4 w-4 text-primary" />
          NeuraShield v2.0.1
        </p>

        {lastAction && (
          <p className="text-sm text-muted-foreground">
            <span className="font-medium">Last Action:</span> {lastAction}
          </p>
        )}

        <p className="text-sm text-muted-foreground flex items-center">
          Powered by xAI & Blockchain
          <Database className="ml-2 h-4 w-4 text-primary" />
        </p>
      </div>
    </footer>
  )
}

