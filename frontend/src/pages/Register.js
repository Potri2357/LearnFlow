import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Auth.css';

import { IconButton } from '@mui/material';
import { 
  Brightness4, 
  Brightness7,
  Person,
  Email,
  Lock,
  AccountCircle,
  School,
  AutoAwesome,
  EmojiObjects,
  TrendingUp,
  Stars,
  Rocket,
  Psychology,
  MenuBook
} from '@mui/icons-material';
import { useColorMode } from '../context/ThemeContext';

export default function Register() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    password2: '',
    first_name: '',
    last_name: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { register, login } = useAuth();
  const { mode, toggleColorMode } = useColorMode();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (formData.password !== formData.password2) {
      setError('Passwords do not match');
      return;
    }
    
    setLoading(true);
    const result = await register(formData);
    
    if (result.success) {
      // Auto-login after registration
      await login(formData.username, formData.password);
      navigate('/upload'); // Redirect to upload page after registration
    } else {
      let errorMessage = 'Registration failed. Please try again.';
      if (typeof result.error === 'string') {
        errorMessage = result.error;
      } else if (result.error && typeof result.error === 'object') {
        // Extract values from the error object
        const values = Object.values(result.error).flat();
        if (values.length > 0) {
          errorMessage = values[0];
        }
      }
      setError(errorMessage);
    }
    
    setLoading(false);
  };

  return (
    <div className="auth-container">
      <div style={{ position: 'absolute', top: 20, right: 20 }}>
        <IconButton 
          onClick={toggleColorMode} 
          sx={{ 
            color: 'white',
            border: '1px solid rgba(255,255,255,0.3)',
            '&:hover': { bgcolor: 'rgba(255,255,255,0.1)' }
          }}
        >
          {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
        </IconButton>
      </div>
      
      {/* Floating Decorative Icons */}
      <div style={{ position: 'absolute', top: '10%', left: '5%', opacity: 0.1, zIndex: 1 }}>
        <AutoAwesome sx={{ fontSize: 80, color: 'white' }} />
      </div>
      <div style={{ position: 'absolute', top: '30%', right: '8%', opacity: 0.1, zIndex: 1 }}>
        <EmojiObjects sx={{ fontSize: 60, color: 'white' }} />
      </div>
      <div style={{ position: 'absolute', bottom: '15%', left: '10%', opacity: 0.1, zIndex: 1 }}>
        <TrendingUp sx={{ fontSize: 70, color: 'white' }} />
      </div>
      <div style={{ position: 'absolute', top: '50%', left: '3%', opacity: 0.08, zIndex: 1 }}>
        <School sx={{ fontSize: 90, color: 'white' }} />
      </div>
      <div style={{ position: 'absolute', top: '70%', right: '5%', opacity: 0.08, zIndex: 1 }}>
        <AutoAwesome sx={{ fontSize: 65, color: 'white', transform: 'rotate(45deg)' }} />
      </div>
      <div style={{ position: 'absolute', top: '20%', left: '50%', opacity: 0.06, zIndex: 1 }}>
        <EmojiObjects sx={{ fontSize: 55, color: 'white', transform: 'rotate(-20deg)' }} />
      </div>
      <div style={{ position: 'absolute', bottom: '30%', right: '12%', opacity: 0.09, zIndex: 1 }}>
        <TrendingUp sx={{ fontSize: 75, color: 'white', transform: 'rotate(15deg)' }} />
      </div>
      <div style={{ position: 'absolute', bottom: '5%', right: '50%', opacity: 0.07, zIndex: 1 }}>
        <School sx={{ fontSize: 50, color: 'white', transform: 'rotate(-30deg)' }} />
      </div>
      <div style={{ position: 'absolute', top: '15%', right: '15%', opacity: 0.09, zIndex: 1 }}>
        <Stars sx={{ fontSize: 55, color: 'white', transform: 'rotate(25deg)' }} />
      </div>
      <div style={{ position: 'absolute', top: '60%', left: '8%', opacity: 0.08, zIndex: 1 }}>
        <Rocket sx={{ fontSize: 70, color: 'white', transform: 'rotate(-15deg)' }} />
      </div>
      <div style={{ position: 'absolute', bottom: '40%', right: '6%', opacity: 0.07, zIndex: 1 }}>
        <Psychology sx={{ fontSize: 60, color: 'white', transform: 'rotate(20deg)' }} />
      </div>
      <div style={{ position: 'absolute', top: '40%', right: '50%', opacity: 0.06, zIndex: 1 }}>
        <MenuBook sx={{ fontSize: 65, color: 'white', transform: 'rotate(-25deg)' }} />
      </div>
      <div style={{ position: 'absolute', bottom: '20%', left: '50%', opacity: 0.08, zIndex: 1 }}>
        <Stars sx={{ fontSize: 45, color: 'white', transform: 'rotate(60deg)' }} />
      </div>
      <div style={{ position: 'absolute', top: '80%', left: '15%', opacity: 0.07, zIndex: 1 }}>
        <Rocket sx={{ fontSize: 50, color: 'white', transform: 'rotate(35deg)' }} />
      </div>
      
      <div className="auth-card enhanced-card">
        {/* Logo */}
        <div className="auth-logo">
          <div className="logo-icon">
            <School sx={{ fontSize: 48, color: '#2563eb' }} />
          </div>
        </div>
        
        <div className="auth-header">
          <h1 className="gradient-text">Create Account</h1>
          <p className="auth-subtitle">Join us to start your learning journey</p>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="first_name">
                <Person sx={{ fontSize: 18, mr: 0.5, verticalAlign: 'middle' }} />
                First Name
              </label>
              <input
                type="text"
                id="first_name"
                name="first_name"
                value={formData.first_name}
                onChange={handleChange}
                className="enhanced-input"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="last_name">
                <Person sx={{ fontSize: 18, mr: 0.5, verticalAlign: 'middle' }} />
                Last Name
              </label>
              <input
                type="text"
                id="last_name"
                name="last_name"
                value={formData.last_name}
                onChange={handleChange}
                className="enhanced-input"
              />
            </div>
          </div>
          
          <div className="form-group">
            <label htmlFor="username">
              <AccountCircle sx={{ fontSize: 18, mr: 0.5, verticalAlign: 'middle' }} />
              Username *
            </label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
              className="enhanced-input"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="email">
              <Email sx={{ fontSize: 18, mr: 0.5, verticalAlign: 'middle' }} />
              Email *
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              className="enhanced-input"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">
              <Lock sx={{ fontSize: 18, mr: 0.5, verticalAlign: 'middle' }} />
              Password *
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              className="enhanced-input"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password2">
              <Lock sx={{ fontSize: 18, mr: 0.5, verticalAlign: 'middle' }} />
              Confirm Password *
            </label>
            <input
              type="password"
              id="password2"
              name="password2"
              value={formData.password2}
              onChange={handleChange}
              required
              className="enhanced-input"
            />
          </div>
          
          <button type="submit" className="btn-primary btn-gradient" disabled={loading}>
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>
        
        <div className="divider">
          <span>or</span>
        </div>
        
        <button className="btn-google" onClick={() => alert('Google OAuth not configured yet')}>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M17.64 9.20443C17.64 8.56625 17.5827 7.95262 17.4764 7.36353H9V10.8449H13.8436C13.635 11.9699 13.0009 12.9231 12.0477 13.5613V15.8194H14.9564C16.6582 14.2526 17.64 11.9453 17.64 9.20443Z" fill="#4285F4"/>
            <path d="M8.99976 18C11.4298 18 13.467 17.1941 14.9561 15.8195L12.0475 13.5613C11.2416 14.1013 10.2107 14.4204 8.99976 14.4204C6.65567 14.4204 4.67158 12.8372 3.96385 10.71H0.957031V13.0418C2.43794 15.9831 5.48158 18 8.99976 18Z" fill="#34A853"/>
            <path d="M3.96409 10.7098C3.78409 10.1698 3.68182 9.59301 3.68182 8.99983C3.68182 8.40665 3.78409 7.82983 3.96409 7.28983V4.95801H0.957273C0.347727 6.17301 0 7.54756 0 8.99983C0 10.4521 0.347727 11.8266 0.957273 13.0416L3.96409 10.7098Z" fill="#FBBC05"/>
            <path d="M8.99976 3.57955C10.3211 3.57955 11.5075 4.03364 12.4402 4.92545L15.0216 2.34409C13.4629 0.891818 11.4257 0 8.99976 0C5.48158 0 2.43794 2.01682 0.957031 4.95818L3.96385 7.29C4.67158 5.16273 6.65567 3.57955 8.99976 3.57955Z" fill="#EA4335"/>
          </svg>
          Continue with Google
        </button>
        
        <p className="auth-footer">
          Already have an account? <Link to="/login" className="link-gradient">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
