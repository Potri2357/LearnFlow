import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Auth.css';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await login(username, password);
    
    if (result.success) {
      navigate('/upload'); // Redirect to upload page after login
    } else {
      setError(result.error);
    }
    
    setLoading(false);
  };

  const handleGoogleLogin = () => {
    // Navigate to beautiful intermediate page
    navigate('/google-login');
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        {/* Logo and Branding */}
        <div className="auth-logo">
          <div className="logo-icon">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <rect width="48" height="48" rx="12" fill="url(#gradient)" />
              <path d="M24 12L32 20L24 28L16 20L24 12Z" fill="white" opacity="0.9" />
              <path d="M24 20L32 28L24 36L16 28L24 20Z" fill="white" opacity="0.7" />
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="48" y2="48">
                  <stop offset="0%" stopColor="#667eea" />
                  <stop offset="100%" stopColor="#764ba2" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h1 className="brand-name">LearnFlow</h1>
        </div>
        
        <h2 className="auth-title">Welcome Back</h2>
        <p className="auth-subtitle">Sign in to continue your learning journey</p>
        
        {error && <div className="error-message">{error}</div>}
        
        {/* Google Sign In Button */}
        <button type="button" className="btn-google" onClick={handleGoogleLogin}>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z" fill="#4285F4"/>
            <path d="M9.003 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.96v2.332C2.44 15.983 5.485 18 9.003 18z" fill="#34A853"/>
            <path d="M3.964 10.712c-.18-.54-.282-1.117-.282-1.71 0-.593.102-1.17.282-1.71V4.96H.957C.347 6.175 0 7.55 0 9.002c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
            <path d="M9.003 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.464.891 11.426 0 9.003 0 5.485 0 2.44 2.017.96 4.958L3.967 7.29c.708-2.127 2.692-3.71 5.036-3.71z" fill="#EA4335"/>
          </svg>
          Continue with Google
        </button>
        
        <div className="divider">
          <span>or</span>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 8a3 3 0 100-6 3 3 0 000 6zm2 1a4 4 0 00-4 0c-1.18.6-2 1.8-2 3.2V14h8v-1.8c0-1.4-.82-2.6-2-3.2z"/>
              </svg>
              Username
            </label>
            <div className="input-with-icon">
              <input
                type="text"
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                required
                autoFocus
              />
            </div>
          </div>
          
          <div className="form-group">
            <label htmlFor="password">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M11 5V4a3 3 0 00-6 0v1H4a1 1 0 00-1 1v6a1 1 0 001 1h8a1 1 0 001-1V6a1 1 0 00-1-1h-1zM6 4a2 2 0 114 0v1H6V4z"/>
              </svg>
              Password
            </label>
            <div className="input-with-icon">
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
              />
            </div>
          </div>
          
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? (
              <>
                <svg className="spinner" width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeDasharray="30" />
                </svg>
                Signing in...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M8 0a8 8 0 100 16A8 8 0 008 0zm3.5 7.5l-4 4a.5.5 0 01-.7 0l-2-2a.5.5 0 11.7-.7L7 10.3l3.6-3.6a.5.5 0 11.7.7z"/>
                </svg>
                Sign In
              </>
            )}
          </button>
        </form>
        
        <p className="auth-footer">
          Don't have an account? <Link to="/register">Sign up</Link>
        </p>
      </div>
    </div>
  );
}
