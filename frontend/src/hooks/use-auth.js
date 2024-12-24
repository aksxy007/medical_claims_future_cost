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
      console.log("Session fetch response:", response.data);

      if (response.status === 200 && response.data?.user && response.data?.accessToken) {
        setUser(response.data.user);
        setToken(response.data.accessToken);
        localStorage.setItem("token", response.data.accessToken);
      } else {
        console.warn("Incomplete session response or unauthorized:", response.data);
        setUser(null);
        setToken(null);
        localStorage.removeItem("token");
      }
    } catch (error) {
      console.error("Error fetching session:", error);
      setUser(null);
      setToken(null);
      localStorage.removeItem("token");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // if (!token) {
    //   setUser(null); // No token means the user is logged out
    //   setLoading(false);
    //   return;
    // }
    // console.group("Fetching User Session");
    fetchUserSession().finally(() => console.groupEnd());
  }, []);

  const login = async (credentials) => {
    // Check if the user is already logged in
    if (user) {
      console.log("User is already logged in, redirecting to dashboard...");
      router.push("/dashboard"); // Redirect to dashboard if user is logged in
      return;
    }

    try {
      const response = await apiClient.post("/auth/login", credentials); // Login request
      if (response.status === 200) {
        setUser(response.data.user);
        setToken(response.data.accessToken);
        localStorage.setItem("token", response.data.accessToken);

        // After successful login, redirect to dashboard or intended page
        // const redirectUrl = localStorage.getItem("redirectAfterLogin") || "/dashboard";
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
        router.push("/"); // Redirect to the home page
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
