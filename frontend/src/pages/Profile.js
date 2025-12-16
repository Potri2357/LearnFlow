import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import './Profile.css';

// Generate a consistent color based on username
const getAvatarColor = (username) => {
  if (!username) return { primary: '#667eea', secondary: '#764ba2' };
  
  const colors = [
    { primary: '#667eea', secondary: '#764ba2' }, // Purple
    { primary: '#f093fb', secondary: '#f5576c' }, // Pink
    { primary: '#4facfe', secondary: '#00f2fe' }, // Blue
    { primary: '#43e97b', secondary: '#38f9d7' }, // Green
    { primary: '#fa709a', secondary: '#fee140' }, // Sunset
    { primary: '#30cfd0', secondary: '#330867' }, // Ocean
    { primary: '#a8edea', secondary: '#fed6e3' }, // Pastel
    { primary: '#ff9a56', secondary: '#ff6a88' }, // Orange
  ];
  
  const hash = username.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors[hash % colors.length];
};

// Generate initials from username
const getInitials = (username, firstName, lastName) => {
  if (firstName && lastName) {
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase();
  }
  if (username) {
    const parts = username.split(/[\s_-]/);
    if (parts.length > 1) {
      return `${parts[0].charAt(0)}${parts[1].charAt(0)}`.toUpperCase();
    }
    return username.substring(0, 2).toUpperCase();
  }
  return 'U';
};

export default function Profile() {
  const { user } = useAuth();
  const [profile, setProfile] = useState(null);
  const [editing, setEditing] = useState(false);
  const [bio, setBio] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [avatarStyle, setAvatarStyle] = useState('gradient'); // 'gradient' or 'dicebear'

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await axios.get('http://localhost:8000/api/profile/', {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      setProfile(response.data);
      setBio(response.data.bio || '');
    } catch (error) {
      console.error('Failed to fetch profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setMessage('');
    
    try {
      const token = localStorage.getItem('access_token');
      await axios.put(
        'http://localhost:8000/api/profile/',
        { bio },
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );
      
      setMessage('Profile updated successfully!');
      setEditing(false);
      fetchProfile();
    } catch (error) {
      setMessage('Failed to update profile');
      console.error('Failed to update profile:', error);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div className="profile-container"><div className="loading">Loading...</div></div>;
  }

  const colors = getAvatarColor(user?.username);
  const initials = getInitials(user?.username, user?.first_name, user?.last_name);
  const dicebearUrl = `https://api.dicebear.com/7.x/initials/svg?seed=${user?.username || 'User'}&backgroundColor=${colors.primary.substring(1)}`;

  return (
    <div className="profile-container">
      <div className="profile-card">
        <div className="profile-header">
          {/* Avatar with style toggle */}
          <div className="avatar-container">
            {avatarStyle === 'gradient' ? (
              <div 
                className="avatar-gradient"
                style={{
                  background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`
                }}
              >
                {initials}
              </div>
            ) : (
              <div className="avatar-dicebear">
                <img src={dicebearUrl} alt="Avatar" />
              </div>
            )}
            
            {/* Avatar style toggle */}
            <button 
              className="avatar-toggle"
              onClick={() => setAvatarStyle(avatarStyle === 'gradient' ? 'dicebear' : 'gradient')}
              title="Change avatar style"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
                <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
              </svg>
            </button>
          </div>

          <h1>{user?.username}</h1>
          <p className="user-email">{user?.email}</p>
          
          {/* Stats cards */}
          <div className="stats-container">
            <div className="stat-card">
              <div className="stat-icon">📚</div>
              <div className="stat-value">{profile?.total_quizzes || 0}</div>
              <div className="stat-label">Quizzes Taken</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🎯</div>
              <div className="stat-value">{profile?.average_score || 0}%</div>
              <div className="stat-label">Avg Score</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🔥</div>
              <div className="stat-value">{profile?.streak_days || 0}</div>
              <div className="stat-label">Day Streak</div>
            </div>
          </div>
        </div>

        <div className="profile-info">
          <div className="info-row">
            <span className="label">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              First Name
            </span>
            <span className="value">{user?.first_name || 'Not set'}</span>
          </div>
          <div className="info-row">
            <span className="label">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              Last Name
            </span>
            <span className="value">{user?.last_name || 'Not set'}</span>
          </div>
          <div className="info-row">
            <span className="label">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="16" y1="2" x2="16" y2="6"></line>
                <line x1="8" y1="2" x2="8" y2="6"></line>
                <line x1="3" y1="10" x2="21" y2="10"></line>
              </svg>
              Member Since
            </span>
            <span className="value">
              {profile?.date_joined ? new Date(profile.date_joined).toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              }) : 'Unknown'}
            </span>
          </div>
        </div>

        <div className="bio-section">
          <div className="section-header">
            <h3>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              About Me
            </h3>
            {!editing && (
              <button className="btn-edit" onClick={() => setEditing(true)}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
                Edit
              </button>
            )}
          </div>
          
          {editing ? (
            <div className="bio-edit">
              <textarea
                value={bio}
                onChange={(e) => setBio(e.target.value)}
                placeholder="Tell us about yourself... What are you learning? What are your goals?"
                rows="5"
              />
              <div className="bio-actions">
                <button className="btn-cancel" onClick={() => {
                  setEditing(false);
                  setBio(profile?.bio || '');
                }}>
                  Cancel
                </button>
                <button className="btn-save" onClick={handleSave} disabled={saving}>
                  {saving ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </div>
          ) : (
            <p className="bio-text">{profile?.bio || 'No bio yet. Click edit to tell us about yourself!'}</p>
          )}
        </div>

        {message && (
          <div className={`message ${message.includes('success') ? 'success' : 'error'}`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
}
