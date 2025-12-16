import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './GoogleCallback.css';

export default function GoogleCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { fetchCurrentUser } = useAuth();
  const [status, setStatus] = useState('processing'); // processing, success, error

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Check if there's an error from Google
        const error = searchParams.get('error');
        if (error) {
          setStatus('error');
          setTimeout(() => navigate('/login'), 3000);
          return;
        }

        // Get JWT tokens from URL params
        const accessToken = searchParams.get('access_token');
        const refreshToken = searchParams.get('refresh_token');

        if (accessToken && refreshToken) {
          // Store tokens in localStorage
          localStorage.setItem('access_token', accessToken);
          localStorage.setItem('refresh_token', refreshToken);
          
          // Fetch the current user to update auth state
          await fetchCurrentUser();
          setStatus('success');
          
          // Redirect to upload page after 2 seconds
          setTimeout(() => navigate('/upload'), 2000);
        } else {
          setStatus('error');
          setTimeout(() => navigate('/login'), 3000);
        }
      } catch (err) {
        console.error('OAuth callback error:', err);
        setStatus('error');
        setTimeout(() => navigate('/login'), 3000);
      }
    };

    handleCallback();
  }, [searchParams, navigate, fetchCurrentUser]);

  return (
    <div className="google-callback-container">
      <div className="callback-card">
        {status === 'processing' && (
          <>
            <div className="spinner-container">
              <div className="spinner"></div>
            </div>
            <h2>Connecting with Google...</h2>
            <p>Please wait while we set up your account</p>
          </>
        )}

        {status === 'success' && (
          <>
            <div className="success-icon">
              <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                <circle cx="40" cy="40" r="40" fill="#10B981" fillOpacity="0.1"/>
                <circle cx="40" cy="40" r="32" fill="#10B981"/>
                <path d="M25 40L35 50L55 30" stroke="white" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <h2>Welcome to LearnFlow! 🎉</h2>
            <p>Your account has been successfully connected</p>
            <p className="redirect-text">Redirecting to dashboard...</p>
          </>
        )}

        {status === 'error' && (
          <>
            <div className="error-icon">
              <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                <circle cx="40" cy="40" r="40" fill="#EF4444" fillOpacity="0.1"/>
                <circle cx="40" cy="40" r="32" fill="#EF4444"/>
                <path d="M30 30L50 50M50 30L30 50" stroke="white" strokeWidth="4" strokeLinecap="round"/>
              </svg>
            </div>
            <h2>Connection Failed</h2>
            <p>There was a problem connecting with Google</p>
            <p className="redirect-text">Redirecting to login...</p>
          </>
        )}
      </div>
    </div>
  );
}
