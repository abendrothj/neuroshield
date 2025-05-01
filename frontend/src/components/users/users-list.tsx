"use client"

import { useState, useEffect } from "react"
import { getUsers } from "@/lib/api/users"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/auth/auth-provider"

export function UsersList({ roleFilter = "all", statusFilter = "all", searchQuery = "" }) {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [userDialogOpen, setUserDialogOpen] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [selectedUser, setSelectedUser] = useState(null)
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    role: "User",
    status: "Active",
  })
  const [saving, setSaving] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [userToDelete, setUserToDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)
  
  const { toast } = useToast()
  const { user } = useAuth()
  
  const isAdmin = user?.role === "admin"

  useEffect(() => {
    async function fetchUsers() {
      try {
        setLoading(true)
        const data = await getUsers()
        
        // Apply filters
        let filteredUsers = data
        
        if (roleFilter !== "all") {
          filteredUsers = filteredUsers.filter(user => 
            user.role.toLowerCase() === roleFilter.toLowerCase()
          )
        }
        
        if (statusFilter !== "all") {
          filteredUsers = filteredUsers.filter(user => 
            user.status.toLowerCase() === statusFilter.toLowerCase()
          )
        }
        
        if (searchQuery) {
          const query = searchQuery.toLowerCase()
          filteredUsers = filteredUsers.filter(user => 
            user.name.toLowerCase().includes(query) || 
            user.email.toLowerCase().includes(query) ||
            user.role.toLowerCase().includes(query)
          )
        }
        
        setUsers(filteredUsers)
        setError(null)
      } catch (err) {
        console.error("Failed to fetch users:", err)
        setError("Failed to load users")
      } finally {
        setLoading(false)
      }
    }

    fetchUsers()
  }, [roleFilter, statusFilter, searchQuery])

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: value,
    })
  }

  const handleSelectChange = (name, value) => {
    setFormData({
      ...formData,
      [name]: value,
    })
  }

  const handleAddUser = () => {
    if (!isAdmin) {
      toast({
        title: "Permission denied",
        description: "Only administrators can add users",
        variant: "destructive",
      })
      return
    }
    
    setIsEditing(false)
    setSelectedUser(null)
    setFormData({
      name: "",
      email: "",
      password: "",
      role: "User",
      status: "Active",
    })
    setUserDialogOpen(true)
  }

  const handleEditUser = (user) => {
    if (!isAdmin) {
      toast({
        title: "Permission denied",
        description: "Only administrators can edit users",
        variant: "destructive",
      })
      return
    }
    
    setIsEditing(true)
    setSelectedUser(user)
    setFormData({
      name: user.name,
      email: user.email,
      password: "", // Don't populate password for security
      role: user.role,
      status: user.status,
    })
    setUserDialogOpen(true)
  }

  const handleDeleteClick = (user) => {
    if (!isAdmin) {
      toast({
        title: "Permission denied",
        description: "Only administrators can delete users",
        variant: "destructive",
      })
      return
    }
    
    setUserToDelete(user)
    setDeleteDialogOpen(true)
  }

  const handleSaveUser = async () => {
    if (!isAdmin) {
      return
    }
    
    // Validate form
    if (!formData.name || !formData.email) {
      toast({
        title: "Validation error",
        description: "Name and email are required",
        variant: "destructive",
      })
      return
    }
