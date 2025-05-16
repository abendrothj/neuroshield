import { Metadata } from "next"
import EventVerification from "@/components/blockchain/event-verification"

export const metadata: Metadata = {
  title: "Event Verification | NeuraShield",
  description: "Verify the authenticity and integrity of security events using blockchain technology",
}

export default function VerificationPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <main className="flex-1">
        <section className="container grid items-center gap-6 pt-6 pb-8 md:py-10">
          <div className="flex max-w-[980px] flex-col items-start gap-2">
            <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl">
              Blockchain Verification
            </h1>
            <p className="text-lg text-muted-foreground">
              Verify the authenticity and integrity of security events recorded on the blockchain
            </p>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <div className="col-span-2">
              <EventVerification />
            </div>
            <div>
              <div className="rounded-lg border bg-card p-6 shadow-sm">
                <h3 className="text-xl font-semibold">How Verification Works</h3>
                <div className="mt-4 space-y-4">
                  <div className="flex items-start">
                    <div className="h-7 w-7 rounded-full bg-blue-100 flex items-center justify-center mr-3 text-blue-600 font-bold">1</div>
                    <div>
                      <h4 className="font-medium">Event Recording</h4>
                      <p className="text-sm text-muted-foreground">Security events are recorded with a cryptographic hash</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="h-7 w-7 rounded-full bg-blue-100 flex items-center justify-center mr-3 text-blue-600 font-bold">2</div>
                    <div>
                      <h4 className="font-medium">Blockchain Storage</h4>
                      <p className="text-sm text-muted-foreground">The event and hash are stored on the blockchain</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="h-7 w-7 rounded-full bg-blue-100 flex items-center justify-center mr-3 text-blue-600 font-bold">3</div>
                    <div>
                      <h4 className="font-medium">Verification</h4>
                      <p className="text-sm text-muted-foreground">Data integrity and blockchain proof are verified</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="h-7 w-7 rounded-full bg-blue-100 flex items-center justify-center mr-3 text-blue-600 font-bold">4</div>
                    <div>
                      <h4 className="font-medium">Certificate</h4>
                      <p className="text-sm text-muted-foreground">A verification certificate is generated as proof</p>
                    </div>
                  </div>
                </div>
                <div className="mt-6">
                  <a 
                    href="/docs/blockchain-verification.pdf" 
                    className="text-sm font-medium text-blue-600 hover:underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Read documentation →
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
} 