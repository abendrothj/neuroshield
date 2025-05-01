import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

export default function ProfilePage() {
  return (
    <div className="space-y-6 ml-64">
      <div>
        <h1 className="text-3xl font-bold">Profile</h1>
        <p className="text-muted-foreground">Manage your personal information and preferences</p>
      </div>

      <div className="grid gap-6 md:grid-cols-[250px_1fr]">
        <Card className="bg-[#0d1117] border-gray-800">
          <CardHeader>
            <CardTitle>Profile Picture</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            <Avatar className="h-32 w-32">
              <AvatarImage src="/placeholder.svg?height=128&width=128" alt="Profile" />
              <AvatarFallback>AU</AvatarFallback>
            </Avatar>
            <div className="mt-4 space-y-2">
              <Button variant="outline" className="w-full border-gray-700">
                Upload New
              </Button>
              <Button variant="outline" className="w-full border-gray-700">
                Remove
              </Button>
            </div>
          </CardContent>
        </Card>

        <Tabs defaultValue="personal" className="w-full">
          <TabsList className="mb-4 bg-[#0f1117]">
            <TabsTrigger value="personal" className="data-[state=active]:bg-[#2a2d3a]">
              Personal Info
            </TabsTrigger>
            <TabsTrigger value="account" className="data-[state=active]:bg-[#2a2d3a]">
              Account
            </TabsTrigger>
            <TabsTrigger value="activity" className="data-[state=active]:bg-[#2a2d3a]">
              Activity
            </TabsTrigger>
          </TabsList>

          <TabsContent value="personal">
            <Card className="bg-[#0d1117] border-gray-800">
              <CardHeader>
                <CardTitle>Personal Information</CardTitle>
                <CardDescription>Update your personal details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="first-name">First Name</Label>
                    <Input id="first-name" defaultValue="Admin" className="bg-[#1a1d29] border-gray-700" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="last-name">Last Name</Label>
                    <Input id="last-name" defaultValue="User" className="bg-[#1a1d29] border-gray-700" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" defaultValue="admin@neurashield.com" className="bg-[#1a1d29] border-gray-700" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <Input id="phone" defaultValue="+1 (555) 123-4567" className="bg-[#1a1d29] border-gray-700" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="job-title">Job Title</Label>
                  <Input
                    id="job-title"
                    defaultValue="Security Administrator"
                    className="bg-[#1a1d29] border-gray-700"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="department">Department</Label>
                  <Input id="department" defaultValue="IT Security" className="bg-[#1a1d29] border-gray-700" />
                </div>
              </CardContent>
              <CardFooter>
                <Button className="bg-blue-500 hover:bg-blue-600">Save Changes</Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="account">
            <Card className="bg-[#0d1117] border-gray-800">
              <CardHeader>
                <CardTitle>Account Information</CardTitle>
                <CardDescription>Manage your account details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input id="username" defaultValue="admin_user" className="bg-[#1a1d29] border-gray-700" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="role">Role</Label>
                  <Input id="role" value="Administrator" readOnly className="bg-[#1a1d29] border-gray-700" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="joined">Joined Date</Label>
                  <Input id="joined" value="January 15, 2023" readOnly className="bg-[#1a1d29] border-gray-700" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="last-login">Last Login</Label>
                  <Input id="last-login" value="Today at 09:42 AM" readOnly className="bg-[#1a1d29] border-gray-700" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="activity">
            <Card className="bg-[#0d1117] border-gray-800">
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>Your recent actions and events</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { action: "Logged in", time: "Today at 09:42 AM", ip: "192.168.1.1" },
                    { action: "Updated threat detection rules", time: "Yesterday at 03:15 PM", ip: "192.168.1.1" },
                    { action: "Generated security report", time: "April 27, 2023 at 11:30 AM", ip: "192.168.1.1" },
                    { action: "Added new API key", time: "April 25, 2023 at 02:45 PM", ip: "192.168.1.1" },
                    { action: "Changed password", time: "April 20, 2023 at 10:15 AM", ip: "192.168.1.1" },
                  ].map((activity, index) => (
                    <div key={index} className="flex justify-between border-b border-gray-800 pb-2">
                      <div>
                        <p className="font-medium">{activity.action}</p>
                        <p className="text-sm text-muted-foreground">{activity.time}</p>
                      </div>
                      <div className="text-sm text-muted-foreground">IP: {activity.ip}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="border-gray-700">
                  View All Activity
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
