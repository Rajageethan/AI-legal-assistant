import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Send, LogOut, MessageSquare, Trash2, Clock, User, Bot } from 'lucide-react';
import { apiClient } from '../../config/api';

const ChatScreen = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const messagesEndRef = useRef(null);
  
  const { currentUser, logout } = useAuth();

  const loadChatHistory = React.useCallback(async () => {
    try {
      const response = await apiClient.get('/api/chat/history');
      
      if (response.data.success) {
        setChatHistory(response.data.history);
      }
    } catch (error) {
      console.warn('Chat history unavailable - backend not connected:', error.message);
      // Continue without chat history
    }
  }, []);

  useEffect(() => {
    loadChatHistory();
    // Add welcome message
    setMessages([{
      id: 'welcome',
      type: 'bot',
      content: "Hello! I'm Clegora. I can help you with questions about Indian law, including the Consumer Protection Act, Right to Information Act, and other legal matters. How can I assist you today?",
      timestamp: new Date().toISOString()
    }]);
  }, [loadChatHistory]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async () => {
    if (!message.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setLoading(true);

    try {
      const response = await apiClient.post('/api/chat', {
        message: userMessage.content,
        conversation_id: currentConversationId,
        save_context: true
      });

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data.response,
        sources: response.data.sources || [],
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Update conversation ID if it's a new conversation
      if (response.data.conversation_id && response.data.conversation_id !== currentConversationId) {
        setCurrentConversationId(response.data.conversation_id);
      }
      
      // Refresh chat history
      loadChatHistory();
    } catch (error) {
      console.error('Error sending message:', error);
      
      let errorContent = "I'm sorry, I'm having trouble connecting to the server right now. Please try again later.";
      
      if (error.message === 'Backend service unavailable') {
        errorContent = "The AI service is currently unavailable. Please check that the backend is deployed and running.";
      }
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: errorContent,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const loadChatContext = async (conversationId) => {
    try {
      const response = await apiClient.get(`/api/chat/context/${conversationId}`);

      if (response.data.success) {
        const context = response.data.context;
        setCurrentConversationId(conversationId);
        
        // Convert context messages to display format
        const contextMessages = [
          {
            id: 'welcome',
            type: 'bot',
            content: "Hello! I'm Clegora. How can I assist you today?",
            timestamp: new Date().toISOString()
          }
        ];
        
        context.messages.forEach((msg, index) => {
          contextMessages.push({
            id: `user-${index}`,
            type: 'user',
            content: msg.user_message,
            timestamp: msg.timestamp
          });
          contextMessages.push({
            id: `bot-${index}`,
            type: 'bot',
            content: msg.bot_response,
            sources: msg.sources || [],
            timestamp: msg.timestamp
          });
        });
        
        setMessages(contextMessages);
      }
    } catch (error) {
      console.error('Error loading chat context:', error);
    }
  };

  const deleteChatContext = async (conversationId) => {
    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }

    try {
      await apiClient.delete(`/api/chat/context/${conversationId}`);
      
      // If this was the current conversation, reset it
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        setMessages([{
          id: 'welcome',
          type: 'bot',
          content: "Hello! I'm Clegora. How can I assist you today?",
          timestamp: new Date().toISOString()
        }]);
      }
      
      await loadChatHistory();
    } catch (error) {
      console.error('Error deleting chat context:', error);
    }
  };

  const startNewChat = () => {
    setCurrentConversationId(null);
    setMessages([{
      id: 'welcome',
      type: 'bot',
      content: "Hello! I'm Clegora. How can I assist you today?",
      timestamp: new Date().toISOString()
    }]);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-900 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white dark:bg-dark-800 border-r border-gray-200 dark:border-dark-700 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold text-gray-800 dark:text-white">Clegora AI</h1>
            <button
              onClick={logout}
              className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
              title="Sign Out"
            >
              <LogOut className="h-5 w-5" />
            </button>
          </div>
          
          <div className="flex items-center text-sm text-gray-600 dark:text-gray-300 mb-4">
            <User className="h-4 w-4 mr-2" />
            {currentUser?.email}
          </div>
          
          <button
            onClick={startNewChat}
            className="w-full py-2 px-4 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors flex items-center justify-center"
          >
            <MessageSquare className="h-4 w-4 mr-2" />
            New Chat
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Recent Conversations</h3>
          
          {chatHistory.length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-8">
              No conversations yet.<br />Start chatting to see your history!
            </p>
          ) : (
            <div className="space-y-2">
              {chatHistory.map((chat) => (
                <div
                  key={chat.conversation_id}
                  className={`p-3 rounded-lg border cursor-pointer transition-all hover:bg-gray-50 dark:hover:bg-dark-700 ${
                    currentConversationId === chat.conversation_id 
                      ? 'bg-primary-50 dark:bg-primary-900/20 border-primary-200 dark:border-primary-700' 
                      : 'bg-white dark:bg-dark-800 border-gray-200 dark:border-dark-600'
                  }`}
                  onClick={() => loadChatContext(chat.conversation_id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-gray-800 dark:text-white truncate">
                        {chat.title}
                      </h4>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 line-clamp-2">
                        {chat.last_message}
                      </p>
                      <div className="flex items-center mt-2 text-xs text-gray-400 dark:text-gray-500">
                        <Clock className="h-3 w-3 mr-1" />
                        {new Date(chat.updated_at).toLocaleDateString()}
                        <span className="mx-2">•</span>
                        {chat.message_count} messages
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteChatContext(chat.conversation_id);
                      }}
                      className="p-1 text-gray-400 dark:text-gray-500 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                      title="Delete conversation"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-2xl px-4 py-3 ${
                  msg.type === 'user'
                    ? 'bg-primary-500 text-white'
                    : 'bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-600 text-gray-800 dark:text-white'
                }`}
              >
                <div className="flex items-start space-x-3">
                  {msg.type === 'bot' && (
                    <div className="flex-shrink-0 w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary-600 dark:text-primary-400" />
                    </div>
                  )}
                  
                  <div className="flex-1">
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      {msg.content}
                    </div>
                    
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg border border-primary-100 dark:border-primary-800">
                        <h4 className="text-sm font-medium text-primary-800 dark:text-primary-300 mb-2">Sources:</h4>
                        <ul className="text-sm text-primary-700 dark:text-primary-400 space-y-1">
                          {msg.sources.map((source, index) => (
                            <li key={index} className="flex items-start">
                              <span className="text-primary-500 dark:text-primary-400 mr-2">•</span>
                              {source}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    <div className={`text-xs mt-2 ${
                      msg.type === 'user' ? 'text-white/70' : 'text-gray-500 dark:text-gray-400'
                    }`}>
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-600 rounded-2xl px-4 py-3">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary-600 dark:text-primary-400" />
                  </div>
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-dark-700 p-6">
          <div className="flex space-x-4">
            <div className="flex-1">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a legal question..."
                className="w-full px-4 py-3 border border-gray-300 dark:border-dark-600 bg-white dark:bg-dark-800 text-gray-900 dark:text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none placeholder-gray-500 dark:placeholder-gray-400"
                rows="1"
                style={{ minHeight: '48px', maxHeight: '120px' }}
                disabled={loading}
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={loading || !message.trim()}
              className="px-6 py-3 bg-primary-500 text-white rounded-xl hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatScreen;
