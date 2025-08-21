// Create axios instance with base configuration
import axios from 'axios';

// API configuration for different environments
const API_CONFIG = {
  // Use environment variable if available, otherwise fallback to localhost for development
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  
  // API endpoints
  ENDPOINTS: {
    AUTH: '/api/auth',
    CHAT: '/api/chat',
    HEALTH: '/api/health'
  }
};

const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add request interceptor for auth token
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('authToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle network errors gracefully
    if (!error.response) {
      console.warn('Network error - backend may not be available:', error.message);
      return Promise.reject(new Error('Backend service unavailable'));
    }
    
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      console.warn('Authentication expired, redirecting to login');
    }
    return Promise.reject(error);
  }
);

export { API_CONFIG, apiClient };
export default apiClient;
