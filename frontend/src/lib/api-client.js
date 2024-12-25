import axios from "axios";


const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000",
  timeout: 10000, // Timeout for requests
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true, // Send cookies with requests (keep it in case of backend needs for cookies)
});

// Request interceptor to include token from localStorage
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle token refresh
// apiClient.interceptors.response.use(
//   (response) => response,
//   async (error) => {
//     const originalRequest = error.config;

//     console.log("Error in refreshTOken",error)
//     if (error.response?.status === 401 && !originalRequest._retry) {
//       originalRequest._retry = false;

//       try {
//         console.log("Token expired, refreshing token...");
//         const refreshResponse = await apiClient.post(
//           `/auth/refresh-token`,
//         );
        
//         console.log("newAccessToken",refreshResponse.data.accessToken)

//         const newAccessToken = refreshResponse.data.accessToken;
//         localStorage.setItem("token", newAccessToken);

//         originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
//         return apiClient(originalRequest); // Retry the original request with new token
//       } catch (refreshError) {
//         console.error("Token refresh failed:", refreshError);
//         localStorage.removeItem("token"); // Remove token from localStorage if refresh fails
//         window.location.href = "/login"; // Redirect to login page
//         return Promise.reject(refreshError);
//       }
//     }

//     return Promise.reject(error);
//   }
// );

export default apiClient;
