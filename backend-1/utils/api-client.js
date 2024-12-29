import axios from 'axios';

// Create a new instance of axios with some default settings
const apiClient = axios.create({
  baseURL: process.env.BASE_API_URL,  // Replace with the API base URL you want to use
  timeout: 5000,                      // Timeout after 5 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient