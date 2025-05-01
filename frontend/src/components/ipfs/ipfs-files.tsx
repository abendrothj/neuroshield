"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, Download, Upload } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { Input } from "@/components/ui/input"
import { useIPFSFiles } from "@/hooks/api/use-ipfs-data"

export function IPFSFiles() {
  const { files, loading, error, uploadFile, downloadFile } = useIPFSFiles()
  const [uploadOpen, setUploadOpen] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const { toast } = useToast()

  const handleDownload = (cid, filename) => {
    try {
      downloadFile(cid, filename)
      toast({
        title: "Download started",
        description: `Downloading ${filename}`,
      })
    } catch (err) {
      console.error("Failed to download file:", err)
      toast({
        title: "Download failed",
        description: "Failed to download file",
        variant: "destructive",
      })
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select a file to upload",
        variant: "destructive",
      })
      return
    }

    try {
      setUploading(true)
      setUploadProgress(0)

      await uploadFile(selectedFile, (progress) => {
        setUploadProgress(progress)
      })

      setUploadOpen(false)
      setSelectedFile(null)

      toast({
        title: "Upload successful",
        description: `${selectedFile.name} has been uploaded to IPFS`,
      })
    } catch (err) {
      console.error("Failed to upload file:", err)
      toast({
        title: "Upload failed",
        description: "Failed to upload file to IPFS",
        variant: "destructive",
      })
    } finally {
      setUploading(false)
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
    <>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Recent Files</h2>
          <div className="flex space-x-2">
            <Button variant="outline" size="sm">
              View All
            </Button>
            <Button className="bg-blue-500 hover:bg-blue-600" size="sm" onClick={() => setUploadOpen(true)}>
              <Upload className="mr-2 h-4 w-4" />
              Upload File
            </Button>
          </div>
        </div>
        <div className="rounded-md border border-gray-800 bg-[#0d1117]">
          <Table>
            <TableHeader className="bg-[#0d1117]">
              <TableRow className="border-gray-800">
                <TableHead>Name</TableHead>
                <TableHead>CID</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Uploaded</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {files.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                    No files found
                  </TableCell>
                </TableRow>
              ) : (
                files.map((file) => (
                  <TableRow key={file.id} className="border-gray-800">
                    <TableCell className="font-medium">{file.name}</TableCell>
                    <TableCell className="font-mono text-xs">{file.cid}</TableCell>
                    <TableCell>{file.size}</TableCell>
                    <TableCell>{file.type}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          file.status === "Pinned" ? "secondary" : file.status === "Uploading" ? "default" : "outline"
                        }
                        className={
                          file.status === "Pinned"
                            ? "bg-green-900/20 text-green-400 border-green-800"
                            : file.status === "Uploading"
                              ? "bg-blue-900/20 text-blue-400 border-blue-800"
                              : "bg-gray-900/20 text-gray-400 border-gray-800"
                        }
                      >
                        {file.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{file.uploaded}</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDownload(file.cid, file.name)}
                        disabled={file.status === "Uploading"}
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
        <DialogContent className="bg-[#0d1117] border-gray-800 text-white">
          <DialogHeader>
            <DialogTitle>Upload File to IPFS</DialogTitle>
            <DialogDescription>Select a file to upload to the IPFS network</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="file-upload" className="text-sm font-medium">
                File
              </label>
              <Input
                id="file-upload"
                type="file"
                onChange={handleFileChange}
                className="bg-[#1a1d29] border-gray-700"
                disabled={uploading}
              />
            </div>

            {uploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Upload Progress</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadOpen(false)} disabled={uploading}>
              Cancel
            </Button>
            <Button
              className="bg-blue-500 hover:bg-blue-600"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
