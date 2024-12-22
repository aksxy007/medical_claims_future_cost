import axios from 'axios';

const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000", // Set your backend API base URL
  timeout: 10000, // Set a timeout for requests (optional)
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials:true
});

// Add a request interceptor (optional)
apiClient.interceptors.request.use(
  (config) => {
    // Example: Add a token to headers if stored in localStorage
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add a response interceptor (optional)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle errors globally
    if (error.response?.status === 401) {
      console.error("Unauthorized! Redirecting to login.");
      // Example: Redirect to login page
    }
    return Promise.reject(error);
  }
);

export default apiClient;
