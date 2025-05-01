"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { useToast } from "@/hooks/use-toast"
import { Loader2 } from "lucide-react"
import { useAuth } from "@/lib/auth/auth-provider"
import { useSettings } from "@/hooks/api/use-settings-data"

export function SettingsForm() {
  const { settings, loading, error, saving, saveSettings } = useSettings()
  const { toast } = useToast()
  const { user } = useAuth()

  const isAdmin = user?.role === "admin"

  const handleInputChange = (e) => {
    const { name, value, type } = e.target

    // Handle numeric inputs
    if (type === "number") {
      saveSettings({
        ...settings,
        [name]: Number.parseInt(value, 10),
      })
    } else {
      saveSettings({
        ...settings,
        [name]: value,
      })
    }
  }

  const handleSwitchChange = (name, checked) => {
    saveSettings({
      ...settings,
      [name]: checked,
    })
  }

  const handleSave = async (e) => {
    e.preventDefault()

    if (!isAdmin) {
      toast({
        title: "Permission denied",
        description: "Only administrators can update settings",
        variant: "destructive",
      })
      return
    }

    try {
      // Validate numeric inputs
      if (settings.alertThreshold < 0 || settings.alertThreshold > 100) {
        throw new Error("Alert threshold must be between 0 and 100")
      }

      if (settings.retentionDays < 1) {
        throw new Error("Retention days must be at least 1")
      }

      await saveSettings(settings)

      toast({
        title: "Settings saved",
        description: "Your settings have been updated successfully",
      })
    } catch (err) {
      console.error("Failed to save settings:", err)
      toast({
        title: "Save failed",
        description: err.message || "Failed to update settings",
        variant: "destructive",
      })
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-900/20 p-4 border border-red-800">
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  return (
    <form onSubmit={handleSave}>
      <Tabs defaultValue="general" className="w-full">
        <TabsList className="mb-4 bg-[#0f1117]">
          <TabsTrigger value="general" className="data-[state=active]:bg-[#2a2d3a]">
            General
          </TabsTrigger>
          <TabsTrigger value="security" className="data-[state=active]:bg-[#2a2d3a]">
            Security
          </TabsTrigger>
          <TabsTrigger value="integrations" className="data-[state=active]:bg-[#2a2d3a]">
            Integrations
          </TabsTrigger>
          <TabsTrigger value="advanced" className="data-[state=active]:bg-[#2a2d3a]">
            Advanced
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>Manage your general system settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="apiEndpoint">API Endpoint</Label>
                <Input
                  id="apiEndpoint"
                  name="apiEndpoint"
                  value={settings?.apiEndpoint || ""}
                  onChange={handleInputChange}
                  className="bg-[#1a1d29] border-gray-700"
                  disabled={!isAdmin}
                />
                <p className="text-xs text-muted-foreground">The base URL for the NeuraShield API</p>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="notificationsEnabled">Notifications</Label>
                  <p className="text-sm text-muted-foreground">Enable system notifications</p>
                </div>
                <Switch
                  id="notificationsEnabled"
                  checked={settings?.notificationsEnabled || false}
                  onCheckedChange={(checked) => handleSwitchChange("notificationsEnabled", checked)}
                  disabled={!isAdmin}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="autoUpdateEnabled">Auto Updates</Label>
                  <p className="text-sm text-muted-foreground">Enable automatic system updates</p>
                </div>
                <Switch
                  id="autoUpdateEnabled"
                  checked={settings?.autoUpdateEnabled || false}
                  onCheckedChange={(checked) => handleSwitchChange("autoUpdateEnabled", checked)}
                  disabled={!isAdmin}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" className="bg-blue-500 hover:bg-blue-600" disabled={saving || !isAdmin}>
                {saving ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Save Changes"
                )}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="security">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>Manage your security preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="alertThreshold">Alert Threshold (%)</Label>
                <Input
                  id="alertThreshold"
                  name="alertThreshold"
                  type="number"
                  min="0"
                  max="100"
                  value={settings?.alertThreshold || 0}
                  onChange={handleInputChange}
                  className="bg-[#1a1d29] border-gray-700"
                  disabled={!isAdmin}
                />
                <p className="text-xs text-muted-foreground">Threshold for triggering security alerts (0-100)</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="retentionDays">Data Retention (days)</Label>
                <Input
                  id="retentionDays"
                  name="retentionDays"
                  type="number"
                  min="1"
                  value={settings?.retentionDays || 30}
                  onChange={handleInputChange}
                  className="bg-[#1a1d29] border-gray-700"
                  disabled={!isAdmin}
                />
                <p className="text-xs text-muted-foreground">Number of days to retain security event data</p>
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" className="bg-blue-500 hover:bg-blue-600" disabled={saving || !isAdmin}>
                {saving ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Save Changes"
                )}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="integrations">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>Integration Settings</CardTitle>
              <CardDescription>Manage external service integrations</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="ipfsGateway">IPFS Gateway</Label>
                <Input
                  id="ipfsGateway"
                  name="ipfsGateway"
                  value={settings?.ipfsGateway || ""}
                  onChange={handleInputChange}
                  className="bg-[#1a1d29] border-gray-700"
                  disabled={!isAdmin}
                />
                <p className="text-xs text-muted-foreground">URL for the IPFS gateway</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="blockchainNetwork">Blockchain Network</Label>
                <Input
                  id="\
