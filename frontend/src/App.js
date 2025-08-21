import React from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import AuthContainer from './components/Auth/AuthContainer';
import ChatScreen from './components/Chat/ChatScreen';
import './index.css';

function AppContent() {
  const { currentUser, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glass p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-white">Loading...</p>
        </div>
      </div>
    );
  }

  return currentUser ? <ChatScreen /> : <AuthContainer />;
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
