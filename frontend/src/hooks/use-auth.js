"use client";

import { useState, createContext, useEffect, useContext, useLayoutEffect } from "react";
import { useRouter } from "next/navigation";
import apiClient from "@/lib/api-client";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Function to fetch user session
  const fetchUserSession = async () => {
    try {
      console.log("Fetching user session...");
      const response = await apiClient.get("/auth/check-session");

      if (response.status === 200) {
        console.log("Session is active. User data:", response.data.user);
        setUser(response.data.user);
        setToken(response.data.accessToken);
      } else {
        console.log("Session expired or user not found. Attempting to refresh token...");
        setToken(null);
      }
    } catch (error) {
      console.error("Error fetching session:", error);
      setUser(null);
      setToken(null);  // Redirect to login if there's an error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUserSession();
  }, []);

  // Function to refresh the token
  // const refreshToken = async () => {
  //   try {
  //     console.log("Attempting to refresh token...");
  //     const response = await apiClient.post("/auth/refresh-token");

  //     if (response.status === 200) {
  //       console.log("Token refreshed successfully. New access token:", response.data.accessToken);
  //       setToken(response.data.accessToken);
  //     } else {
  //       console.log("Token refresh failed with status:", response.status);
  //       setUser(null);
  //       setToken(null);
  //     }
  //   } catch (error) {
  //     console.error("Error refreshing token:", error);
  //     setUser(null);
  //     setToken(null);
  //   }
  // };

  // Initialize the interceptor and set Authorization headers
  useLayoutEffect(() => {
    const requestInterceptor = apiClient.interceptors.request.use(
      (config) => {
        if (!config._retry && token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Cleanup interceptors on unmount
    return () => {
      apiClient.interceptors.request.eject(requestInterceptor)
    };
  }, [token]); // Re-run effect if the token changes


  useLayoutEffect(()=>{
    const responseInterceptor = apiClient.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 403 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            console.log("Attempting to refresh token inside interceptor...");
            const refreshResponse = await apiClient.post("/auth/refresh-token");
            
            if (refreshResponse?.status === 200) {
              const newAccessToken = refreshResponse.data.accessToken;
              setToken(newAccessToken);
              originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
              return apiClient(originalRequest); // Retry the original request with the new token
            } else {
              console.error("Token refresh failed in interceptor.");
              setUser(null);
              setToken(null);
              router.push("/login"); // Redirect to login if refresh fails
              return Promise.reject(error);
            }
          } catch (refreshError) {
            console.error("Error refreshing token in interceptor:", refreshError);
            setUser(null);
            setToken(null);
            router.push("/login");
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );

    return ()=>{
        apiClient.interceptors.response.eject(responseInterceptor)
    }
  },[router,token])
  // Use effect to fetch the user session initially

  // Login function
  const login = async (credentials) => {
    try {
      const response = await apiClient.post("/auth/login", credentials);

      if (response.status === 200) {
        setUser(response.data.user);
        setToken(response.data.accessToken);
        router.push("/dashboard");
      } else {
        alert("Login failed. Please check your credentials.");
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("An error occurred during login. Please try again.");
    }
  };

  // Logout function
  const logout = async () => {
    try {
      const response = await apiClient.post("/auth/logout");

      if (response.status === 200) {
        setUser(null);
        setToken(null);
        router.push("/");
      } else {
        alert(response.data.message || "Logout failed. Please try again.");
      }
    } catch (error) {
      console.error("Logout error:", error);
      alert("An error occurred during logout. Please try again.");
    }
  };

  // Check if the user is authenticated
  const isAuthenticated = () => token !== null;

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        login,
        logout,
        isAuthenticated
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
