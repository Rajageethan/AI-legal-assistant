import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getAnalytics } from 'firebase/analytics';

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY || "AIzaSyA3zW4kZ_dT5jIKPI4qI5FyqgXb0ZSVxAM",
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || "clegora.firebaseapp.com",
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || "clegora",
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || "clegora.firebasestorage.app",
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || "325502068501",
  appId: process.env.REACT_APP_FIREBASE_APP_ID || "1:325502068501:web:21d8d0ac71f017b22deb1c",
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID || "G-JT7PG2KS5T"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const analytics = getAnalytics(app);
export default app;
