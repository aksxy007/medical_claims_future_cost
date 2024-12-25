"use client";

import { useState, createContext, useEffect, useContext } from "react";
import { useRouter } from "next/navigation";
import apiClient from "@/lib/api-client";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const fetchUserSession = async () => {
    try {
      const response = await apiClient.get("/auth/session");
      console.log(response)
      if (response.status === 200) {
        setUser(response.data.user);
        setToken(response.data.accessToken);
        localStorage.setItem("token",response.data.accessToken)
      }
      else{
        setUser(null)
        setToken(null)
        router.push("/")
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log("Session expired or unauthorized, redirecting to login.");
        setUser(null);
        setToken(null);
        localStorage.removeItem("token");
        // router.push("/login");
      } else {
        console.error("Error fetching session:", error);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // if(!token)
      fetchUserSession();
  }, []);

  const login = async (credentials) => {
    if (user) {
      console.log("User is already logged in, redirecting to dashboard...");
      router.push("/dashboard");
      return;
    }

    try {
      const response = await apiClient.post("/auth/login", credentials);
      if (response.status === 200) {
        setUser(response.data.user);
        setToken(response.data.accessToken);
        localStorage.setItem("token", response.data.accessToken);
        router.push("/dashboard");
      } else {
        console.error("Login failed with status:", response.status);
        alert("Login failed. Please check your credentials.");
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("An error occurred during login. Please try again.");
    }
  };

  const logout = async () => {
    try {
      const response = await apiClient.post("/auth/logout");
      if (response.status === 200) {
        setUser(null);
        setToken(null);
        localStorage.removeItem("token");
        router.push("/");
      } else {
        console.error("Logout failed with status:", response.status, response.data);
        alert(response.data.message || "Logout failed. Please try again.");
      }
    } catch (error) {
      console.error("Logout error:", error);
      alert(error.response?.data?.message || "An error occurred during logout. Please try again.");
    }
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
