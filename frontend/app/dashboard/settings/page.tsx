import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

export default function SettingsPage() {
  return (
    <div className="space-y-6 ml-64">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">Manage your account settings and preferences</p>
      </div>

      <Tabs defaultValue="general" className="w-full">
        <TabsList className="mb-4 bg-[#0f1117]">
          <TabsTrigger value="general" className="data-[state=active]:bg-[#2a2d3a]">
            General
          </TabsTrigger>
          <TabsTrigger value="security" className="data-[state=active]:bg-[#2a2d3a]">
            Security
          </TabsTrigger>
          <TabsTrigger value="notifications" className="data-[state=active]:bg-[#2a2d3a]">
            Notifications
          </TabsTrigger>
          <TabsTrigger value="api" className="data-[state=active]:bg-[#2a2d3a]">
            API
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>Manage your general account settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input id="name" defaultValue="Admin User" className="bg-[#1a1d29] border-gray-700" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" defaultValue="admin@neurashield.com" className="bg-[#1a1d29] border-gray-700" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="timezone">Timezone</Label>
                <select id="timezone" className="w-full rounded-md border border-gray-700 bg-[#1a1d29] p-2">
                  <option>UTC-8 (Pacific Time)</option>
                  <option>UTC-5 (Eastern Time)</option>
                  <option>UTC+0 (GMT)</option>
                  <option>UTC+1 (Central European Time)</option>
                  <option>UTC+8 (China Standard Time)</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="dark-mode">Dark Mode</Label>
                  <p className="text-sm text-muted-foreground">Enable dark mode for the interface</p>
                </div>
                <Switch id="dark-mode" defaultChecked />
              </div>
            </CardContent>
            <CardFooter>
              <Button className="bg-blue-500 hover:bg-blue-600">Save Changes</Button>
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
                <Label htmlFor="current-password">Current Password</Label>
                <Input id="current-password" type="password" className="bg-[#1a1d29] border-gray-700" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-password">New Password</Label>
                <Input id="new-password" type="password" className="bg-[#1a1d29] border-gray-700" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirm-password">Confirm New Password</Label>
                <Input id="confirm-password" type="password" className="bg-[#1a1d29] border-gray-700" />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="2fa">Two-Factor Authentication</Label>
                  <p className="text-sm text-muted-foreground">Enable 2FA for additional security</p>
                </div>
                <Switch id="2fa" />
              </div>
            </CardContent>
            <CardFooter>
              <Button className="bg-blue-500 hover:bg-blue-600">Update Password</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="notifications">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>Notification Settings</CardTitle>
              <CardDescription>Manage how you receive notifications</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Email Notifications</Label>
                  <p className="text-sm text-muted-foreground">Receive notifications via email</p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Critical Alerts</Label>
                  <p className="text-sm text-muted-foreground">Get notified for critical security events</p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>System Updates</Label>
                  <p className="text-sm text-muted-foreground">Receive notifications about system updates</p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Weekly Reports</Label>
                  <p className="text-sm text-muted-foreground">Receive weekly summary reports</p>
                </div>
                <Switch />
              </div>
            </CardContent>
            <CardFooter>
              <Button className="bg-blue-500 hover:bg-blue-600">Save Preferences</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="api">
          <Card className="bg-[#0d1117] border-gray-800">
            <CardHeader>
              <CardTitle>API Settings</CardTitle>
              <CardDescription>Manage your API keys and access</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>API Key</Label>
                <div className="flex">
                  <Input
                    value="sk_live_51NcGJKLkjHGFDSa7890ZXCVBnm"
                    readOnly
                    className="bg-[#1a1d29] border-gray-700"
                  />
                  <Button variant="outline" className="ml-2">
                    Copy
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground">
                  Your API key provides full access to your account. Keep it secure.
                </p>
              </div>
              <div className="space-y-2">
                <Label>Webhook URL</Label>
                <Input placeholder="https://your-domain.com/webhook" className="bg-[#1a1d29] border-gray-700" />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>API Access</Label>
                  <p className="text-sm text-muted-foreground">Enable API access for integrations</p>
                </div>
                <Switch defaultChecked />
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" className="border-gray-700">
                Regenerate API Key
              </Button>
              <Button className="bg-blue-500 hover:bg-blue-600">Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
