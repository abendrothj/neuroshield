"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Save, Eye, EyeOff, RefreshCw, Check } from "lucide-react"
import { Badge } from "@/components/ui/badge"

export default function SettingsView() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [showApiKey, setShowApiKey] = useState(false)

  // Security settings
  const [autoResponse, setAutoResponse] = useState(false)
  const [threatSensitivity, setThreatSensitivity] = useState([50])
  const [enableNotifications, setEnableNotifications] = useState(true)

  // API settings
  const [apiKey, setApiKey] = useState("sk_live_51NzUBtKLksdJFKDJFKDJF")
  const [webhookUrl, setWebhookUrl] = useState("https://api.example.com/webhook")
  const [saveSuccess, setSaveSuccess] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  const handleSave = () => {
    setIsSaving(true)
    // Simulate saving
    setTimeout(() => {
      setIsSaving(false)
      setSaveSuccess(true)

      setTimeout(() => {
        setSaveSuccess(false)
      }, 3000)
    }, 1000)
  }

  const regenerateApiKey = () => {
    // Simulate API key regeneration
    setApiKey("sk_live_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15))
  }

  return (
    <div className={`space-y-6 transition-opacity duration-500 ${isLoaded ? "opacity-100" : "opacity-0"}`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <h1 className="text-2xl font-bold">Settings</h1>
        <Button
          onClick={handleSave}
          disabled={isSaving}
          className="bg-primary hover:bg-primary/80 text-primary-foreground transition-all duration-300"
        >
          {isSaving ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Saving...
            </>
          ) : saveSuccess ? (
            <>
              <Check className="mr-2 h-4 w-4" />
              Saved!
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save Changes
            </>
          )}
        </Button>
      </div>

      <Tabs defaultValue="security" className="w-full">
        <TabsList className="w-full bg-muted/50 border border-white/10">
          <TabsTrigger
            value="security"
            className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            Security
          </TabsTrigger>
          <TabsTrigger
            value="api"
            className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            API & Integrations
          </TabsTrigger>
          <TabsTrigger
            value="account"
            className="flex-1 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            Account
          </TabsTrigger>
        </TabsList>

        <TabsContent value="security" className="mt-4">
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="text-xl">Security Settings</CardTitle>
              <CardDescription>Configure how NeuraShield protects your systems</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="auto-response" className="text-sm">
                    Auto-Response Mode
                  </Label>
                  <p className="text-xs text-muted-foreground">AI will respond automatically to threats</p>
                </div>
                <Switch
                  id="auto-response"
                  checked={autoResponse}
                  onCheckedChange={setAutoResponse}
                  className="data-[state=checked]:bg-primary"
                />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <Label htmlFor="threat-sensitivity" className="text-sm">
                    Threat Sensitivity
                  </Label>
                  <span className="text-sm text-primary">{threatSensitivity}%</span>
                </div>
                <Slider
                  id="threat-sensitivity"
                  defaultValue={[50]}
                  max={100}
                  step={1}
                  value={threatSensitivity}
                  onValueChange={setThreatSensitivity}
                  className="[&>span]:bg-primary"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Low</span>
                  <span>High</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="notifications" className="text-sm">
                    Security Notifications
                  </Label>
                  <p className="text-xs text-muted-foreground">Receive alerts for critical security events</p>
                </div>
                <Switch
                  id="notifications"
                  checked={enableNotifications}
                  onCheckedChange={setEnableNotifications}
                  className="data-[state=checked]:bg-primary"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="mt-4">
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="text-xl">API & Integrations</CardTitle>
              <CardDescription>Manage API keys and third-party integrations</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="api-key" className="text-sm">
                  API Key
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      id="api-key"
                      type={showApiKey ? "text" : "password"}
                      value={showApiKey ? apiKey : "•".repeat(Math.min(20, apiKey.length))}
                      className="bg-muted/50 border-white/10 focus-visible:ring-primary pr-10"
                      readOnly
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-primary"
                      onClick={() => setShowApiKey(!showApiKey)}
                    >
                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    className="border-white/10 text-muted-foreground hover:text-primary hover:border-primary"
                    onClick={regenerateApiKey}
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Regenerate
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="webhook-url" className="text-sm">
                  Webhook URL
                </Label>
                <Input
                  id="webhook-url"
                  value={webhookUrl}
                  onChange={(e) => setWebhookUrl(e.target.value)}
                  className="bg-muted/50 border-white/10 focus-visible:ring-primary"
                />
              </div>

              <div className="pt-4">
                <h3 className="text-sm font-medium mb-3">Connected Services</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 rounded-md bg-muted/30 border border-white/10">
                    <div className="flex items-center gap-2">
                      <div className="h-8 w-8 rounded-full bg-muted/50 flex items-center justify-center">
                        <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path
                            d="M12 2L2 7L12 12L22 7L12 2Z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M2 17L12 22L22 17"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M2 12L12 17L22 12"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </div>
                      <div>
                        <div className="font-medium">Blockchain Network</div>
                        <div className="text-xs text-muted-foreground">Last synced: 2 minutes ago</div>
                      </div>
                    </div>
                    <Badge className="bg-green-500/20 text-green-400 hover:bg-green-500/30">Connected</Badge>
                  </div>
                  <div className="flex justify-between items-center p-3 rounded-md bg-muted/30 border border-white/10">
                    <div className="flex items-center gap-2">
                      <div className="h-8 w-8 rounded-full bg-muted/50 flex items-center justify-center">
                        <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path
                            d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z"
                            fill="currentColor"
                          />
                          <path
                            d="M12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17Z"
                            fill="currentColor"
                          />
                        </svg>
                      </div>
                      <div>
                        <div className="font-medium">AI Engine</div>
                        <div className="text-xs text-muted-foreground">Version: 2.4.1</div>
                      </div>
                    </div>
                    <Badge className="bg-green-500/20 text-green-400 hover:bg-green-500/30">Connected</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="account" className="mt-4">
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="text-xl">Account Settings</CardTitle>
              <CardDescription>Manage your account preferences</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-4 text-muted-foreground text-center">
                <p>Account settings will appear here.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

