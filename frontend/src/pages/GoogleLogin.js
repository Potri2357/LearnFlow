import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './GoogleLogin.css';

export default function GoogleLogin() {
  const navigate = useNavigate();

  useEffect(() => {
    // Auto-redirect to Google OAuth after 2 seconds
    const timer = setTimeout(() => {
      window.location.href = 'http://localhost:8000/accounts/google/login/';
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  const handleManualRedirect = () => {
    window.location.href = 'http://localhost:8000/accounts/google/login/';
  };

  return (
    <div className="google-login-container">
      <div className="google-login-card">
        {/* LearnFlow Logo */}
        <div className="login-logo">
          <div className="logo-icon-large">
            <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
              <rect width="80" height="80" rx="20" fill="url(#gradientLarge)" />
              <path d="M40 20L53.33 33.33L40 46.66L26.67 33.33L40 20Z" fill="white" opacity="0.9" />
              <path d="M40 33.33L53.33 46.66L40 60L26.67 46.66L40 33.33Z" fill="white" opacity="0.7" />
              <defs>
                <linearGradient id="gradientLarge" x1="0" y1="0" x2="80" y2="80">
                  <stop offset="0%" stopColor="#667eea" />
                  <stop offset="100%" stopColor="#764ba2" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h1 className="brand-title">LearnFlow</h1>
          <p className="brand-tagline">Personalized Learning Platform</p>
        </div>

        {/* Connecting Animation */}
        <div className="connecting-section">
          <div className="connection-animation">
            <div className="dot dot-1"></div>
            <div className="line line-1"></div>
            <div className="dot dot-2"></div>
            <div className="line line-2"></div>
            <div className="dot dot-3"></div>
          </div>
          
          <h2 className="connecting-title">Connecting to Google</h2>
          <p className="connecting-subtitle">You'll be redirected to Google's secure login page</p>
        </div>

        {/* Google Logo */}
        <div className="google-logo-section">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <path d="M47.28 24.52c0-1.7-.15-3.34-.44-4.92H24v9.28h13.12c-.56 3-2.25 5.54-4.8 7.24v6.04h7.76c4.54-4.18 7.16-10.34 7.16-17.64z" fill="#4285F4"/>
            <path d="M24.01 48c6.48 0 11.92-2.15 15.89-5.84l-7.76-6.04c-2.15 1.44-4.9 2.3-8.13 2.3-6.25 0-11.55-4.22-13.44-9.9H2.56v6.22C6.5 42.62 14.62 48 24.01 48z" fill="#34A853"/>
            <path d="M10.57 28.52c-.48-1.44-.75-2.98-.75-4.56s.27-3.12.75-4.56V13.18H2.56C.93 16.44 0 20.12 0 24.01c0 3.89.93 7.57 2.56 10.83l8.01-6.32z" fill="#FBBC05"/>
            <path d="M24.01 9.54c3.52 0 6.69 1.21 9.17 3.59l6.88-6.88C35.89 2.38 30.48 0 24.01 0 14.62 0 6.5 5.38 2.56 13.18l8.01 6.22c1.89-5.68 7.19-9.86 13.44-9.86z" fill="#EA4335"/>
          </svg>
        </div>

        {/* Manual Button (backup) */}
        <button className="manual-continue-btn" onClick={handleManualRedirect}>
          <span>Continue Manually</span>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path d="M10 0l10 10-10 10V0z"/>
          </svg>
        </button>

        {/* Security Note */}
        <div className="security-note">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0L2 3v4c0 3.5 2.5 6.5 6 7 3.5-.5 6-3.5 6-7V3L8 0zm0 10.5l-3-3 1-1 2 2 4-4 1 1-5 5z"/>
          </svg>
          <span>Secure authentication powered by Google</span>
        </div>

        {/* Back to Login */}
        <button className="back-link" onClick={() => navigate('/login')}>
          ← Back to Login
        </button>
      </div>
    </div>
  );
}
