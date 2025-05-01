/**
 * Users Data API Hooks
 *
 * This file contains React hooks for interacting with user-related API endpoints.
 * These hooks handle data fetching, caching, and user management operations.
 */

"use client"

import { useState, useEffect, useCallback } from "react"
import {
  getUsers,
  getUserById,
  getCurrentUser,
  createUser,
  updateUser,
  deleteUser,
  type User,
  type CreateUserData,
  type UpdateUserData,
} from "@/lib/api/users"

/**
 * Hook for fetching and managing users with filtering options
 *
 * @param roleFilter - Filter users by role
 * @param statusFilter - Filter users by status
 * @param searchQuery - Search query to filter users
 * @returns Object containing users array, loading state, error state, refetch function, and user management functions
 *
 * @example
 * const { users, loading, error, addUser, editUser, removeUser } = useUsers("admin", "active");
 */
export function useUsers(roleFilter?: string, statusFilter?: string, searchQuery?: string) {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getUsers()

      // Apply filters
      let filteredUsers = data

      if (roleFilter && roleFilter !== "all") {
        filteredUsers = filteredUsers.filter((user) => user.role.toLowerCase() === roleFilter.toLowerCase())
      }

      if (statusFilter && statusFilter !== "all") {
        filteredUsers = filteredUsers.filter((user) => user.status.toLowerCase() === statusFilter.toLowerCase())
      }

      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        filteredUsers = filteredUsers.filter(
          (user) =>
            user.name.toLowerCase().includes(query) ||
            user.email.toLowerCase().includes(query) ||
            user.role.toLowerCase().includes(query),
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
  }, [roleFilter, statusFilter, searchQuery])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  /**
   * Add a new user
   *
   * @param userData - User data to create
   * @returns The newly created user
   */
  const addUser = useCallback(async (userData: CreateUserData) => {
    try {
      const newUser = await createUser(userData)
      // Add the new user to the local state
      setUsers((prev) => [...prev, newUser])
      return newUser
    } catch (err) {
      console.error("Failed to create user:", err)
      throw err
    }
  }, [])

  /**
   * Edit an existing user
   *
   * @param id - The ID of the user to update
   * @param userData - User data to update
   * @returns The updated user
   */
  const editUser = useCallback(async (id: string, userData: UpdateUserData) => {
    try {
      const updatedUser = await updateUser(id, userData)
      // Update the user in the local state
      setUsers((prev) => prev.map((user) => (user.id === id ? updatedUser : user)))
      return updatedUser
    } catch (err) {
      console.error("Failed to update user:", err)
      throw err
    }
  }, [])

  /**
   * Remove a user
   *
   * @param id - The ID of the user to delete
   * @returns Result object with success status
   */
  const removeUser = useCallback(async (id: string) => {
    try {
      const result = await deleteUser(id)
      if (result.success) {
        // Remove the user from the local state
        setUsers((prev) => prev.filter((user) => user.id !== id))
      }
      return result
    } catch (err) {
      console.error("Failed to delete user:", err)
      throw err
    }
  }, [])

  return {
    users,
    loading,
    error,
    refetch: fetchUsers,
    addUser,
    editUser,
    removeUser,
  }
}

/**
 * Hook for fetching details of a specific user
 *
 * @param id - The ID of the user to fetch
 * @returns Object containing user data, loading state, error state, and fetch function
 *
 * @example
 * const { user, loading, error } = useUserDetails("user-123");
 */
export function useUserDetails(id?: string) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchUser = useCallback(async (userId: string) => {
    if (!userId) return

    try {
      setLoading(true)
      const data = await getUserById(userId)
      setUser(data)
      setError(null)
      return data
    } catch (err) {
      console.error("Failed to fetch user details:", err)
      setError("Failed to load user details")
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (id) {
      fetchUser(id)
    }
  }, [id, fetchUser])

  return { user, loading, error, fetchUser }
}

/**
 * Hook for fetching and managing the current logged-in user
 *
 * @returns Object containing user data, loading state, error state, and refetch function
 *
 * @example
 * const { user, loading, error } = useCurrentUser();
 */
export function useCurrentUser() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCurrentUser = useCallback(async () => {
    try {
      setLoading(true)
      const data = await getCurrentUser()
      setUser(data)
      setError(null)
      return data
    } catch (err) {
      console.error("Failed to fetch current user:", err)
      setError("Failed to load user profile")
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCurrentUser()
  }, [fetchCurrentUser])

  return { user, loading, error, refetch: fetchCurrentUser }
}
