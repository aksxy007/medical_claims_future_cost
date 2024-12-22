"use client"

import { useState,createContext,useEffect,useContext } from "react";
import { useRouter } from "next/navigation";
import apiClient from "@/lib/apiClient";

const AuthContext = createContext();

export const AuthProvider = ({children})=>{
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)
    const router = useRouter()

    const fetchUserSession = async () => {
        try {
          const response = await apiClient.get("/auth/session"); // Session endpoint
          setUser(response.data.user); // Update user state with session data
        } catch (error) {
          setUser(null); // No session or error occurred
        } finally {
          setLoading(false); // Stop loading after fetching session
        }
      };

      useEffect(() => {
        fetchUserSession(); // Fetch session on initial render
      }, []);

    const logout = async () => {
        try {
          await apiClient.post("/auth/logout"); // Logout endpoint
          setUser(null); // Clear user state
          router.push("/login"); // Redirect to login page
        } catch (error) {
          console.error("Logout failed:", error);
        }
      };


      const login = async (credentials) => {
        try {
          const response = await apiClient.post("/auth/login", credentials); // Login endpoint
          setUser(response.data.user); // Set user from login response
          router.push("/dashboard"); // Redirect after successful login
        } catch (error) {
          console.error("Login failed:", error);
        }
      };
    
      return <AuthContext.Provider value={{user,loading,login,logout}}>
        {children}
      </AuthContext.Provider>
}

export const useAuth = ()=>useContext(AuthContext);