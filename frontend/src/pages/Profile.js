import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import API from '../api/api';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Avatar,
  Button,
  TextField,
  MenuItem,
  Chip,
  Switch,
  IconButton,
  Divider,
  useTheme,
  LinearProgress,
  Badge
} from '@mui/material';
import {
  Edit as EditIcon,
  Share as ShareIcon,
  PhotoCamera as PhotoCameraIcon,
  School as SchoolIcon,
  EmojiEvents as EmojiEventsIcon,
  MilitaryTech as MilitaryTechIcon,
  Psychology as PsychologyIcon,
  RocketLaunch as RocketLaunchIcon,
  Lock as LockIcon,
  Badge as BadgeIcon,
  AutoAwesome as AutoAwesomeIcon,
  Close as CloseIcon,
  Add as AddIcon,
  NotificationsActive as NotificationsActiveIcon,
  Schedule as ScheduleIcon,
  Mail as MailIcon,
  Link as LinkIcon,
  GitHub as GitHubIcon,
  Google as GoogleIcon,
  Microsoft as MicrosoftIcon
} from '@mui/icons-material';

export default function Profile() {
  const { user } = useAuth();
  const theme = useTheme();
  
  // State for profile data
  const [profile, setProfile] = useState({
    bio: '',
    school: 'Westside High School',
    grade: '11th Grade',
    subjects: ['Mathematics', 'Physics'],
    preferences: {
        adaptiveDifficulty: true,
        strictMode: false,
        studyReminders: true,
        achievementUnlocks: true,
        weeklyReport: false
    }
  });

  const [stats, setStats] = useState({
      total_quizzes: 0,
      average_score: 0,
      streak_days: 0
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const { data } = await API.get('profile/');
      setProfile({
          bio: data.bio || '',
          school: data.school || '',
          grade: data.grade || '',
          subjects: data.subjects || [],
          preferences: data.preferences || {
              adaptiveDifficulty: true,
              strictMode: false,
              studyReminders: true,
              achievementUnlocks: true,
              weeklyReport: false
          }
      });
      setStats({
          total_quizzes: data.total_quizzes || 0,
          average_score: data.average_score || 0,
          streak_days: data.streak_days || 0
      });
    } catch (error) {
      console.error('Failed to fetch profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await API.put('profile/', {
          bio: profile.bio,
          school: profile.school,
          grade: profile.grade,
          subjects: profile.subjects,
          preferences: profile.preferences
      });
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (field, value) => {
    setProfile(prev => ({ ...prev, [field]: value }));
  };

  const handlePreferenceChange = (key) => {
      setProfile(prev => ({
          ...prev,
          preferences: { ...prev.preferences, [key]: !prev.preferences[key] }
      }));
  };

  return (
    <Box sx={{ pb: 12 }}>
      {/* Header Info */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h2" fontWeight={900} color="text.primary" sx={{ letterSpacing: '-0.02em', mb: 1 }}>My Profile</Typography>
        <Typography variant="body1" color="text.secondary" fontWeight={500}>Manage your personal information, achievements, and account settings.</Typography>
      </Box>

      {/* Banner & Avatar Card */}
      <Paper elevation={0} sx={{ borderRadius: '16px', overflow: 'hidden', border: '1px solid', borderColor: 'divider', bgcolor: 'background.paper', mb: 6 }}>
        <Box sx={{ 
            height: 140, 
            background: `linear-gradient(135deg, ${theme.palette.primary.main}30, ${theme.palette.background.paper} 80%)`,
            borderBottom: '1px solid', borderColor: 'divider'
        }} />
        <Box sx={{ px: 4, pb: 4, mt: -6 }}>
            <Grid container spacing={4} alignItems="flex-start">
                <Grid item>
                    <Box sx={{ position: 'relative' }}>
                        <Avatar 
                            src={`https://api.dicebear.com/7.x/initials/svg?seed=${user?.username}&backgroundColor=${theme.palette.primary.main.slice(1)}`} 
                            sx={{ width: 128, height: 128, border: `4px solid ${theme.palette.background.paper}`, boxShadow: theme.shadows[3] }}
                        />
                        <IconButton 
                            size="small"
                            sx={{ 
                                position: 'absolute', bottom: 4, right: 4, 
                                bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider',
                                '&:hover': { bgcolor: 'action.hover' }
                            }}
                        >
                            <PhotoCameraIcon fontSize="small" />
                        </IconButton>
                    </Box>
                </Grid>
                <Grid item xs>
                    <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', alignItems: { md: 'flex-start' }, gap: 2, pt: { xs: 0, md: 7 } }}>
                        <Box>
                            <Typography variant="h4" fontWeight={800}>{user?.first_name ? `${user.first_name} ${user.last_name}` : user?.username}</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 1, mb: 2, flexWrap: 'wrap' }}>
                                <Typography variant="body2" fontWeight={500} color="text.secondary">Student at {profile.school}</Typography>
                                <Box sx={{ width: 4, height: 4, borderRadius: '50%', bgcolor: 'text.secondary' }} />
                                <Typography variant="body2" color="text.secondary">{profile.grade}</Typography>
                                <Chip label="PRO" size="small" sx={{ height: 20, fontSize: '0.65rem', fontWeight: 800, bgcolor: 'primary.main', color: 'white' }} />
                            </Box>
                            <Paper variant="outlined" sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.05)', maxWidth: 600 }}>
                                <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                                    "{profile.bio || "No bio set yet."}"
                                </Typography>
                            </Paper>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 2, mt: { xs: 2, md: 0 } }}>
                            <Button variant="outlined" startIcon={<ShareIcon />} sx={{ borderRadius: '12px', fontWeight: 700, textTransform: 'none', borderColor: 'divider', color: 'text.primary', '&:hover': { borderColor: 'primary.main', bgcolor: 'rgba(19, 127, 236, 0.05)' } }}>Share Profile</Button>
                            <Button variant="contained" startIcon={<EditIcon />} sx={{ borderRadius: '12px', fontWeight: 700, textTransform: 'none', boxShadow: 'none' }}>Edit Bio</Button>
                        </Box>
                    </Box>
                </Grid>
            </Grid>
        </Box>
      </Paper>

      {/* Stats Row */}
      <Grid container spacing={3} sx={{ mb: 6 }}>
          {[
              { label: 'Quizzes Done', value: stats.total_quizzes, unit: '', color: 'primary.main', bg: 'rgba(19,127,236,0.08)' },
              { label: 'Avg. Score', value: `${stats.average_score}%`, unit: '', color: '#10B981', bg: 'rgba(16,185,129,0.08)' },
              { label: 'Study Streak', value: stats.streak_days, unit: ' days', color: '#F59E0B', bg: 'rgba(245,158,11,0.08)' },
          ].map(s => (
              <Grid item xs={12} sm={4} key={s.label}>
                  <Paper elevation={0} sx={{ p: 3, borderRadius: '16px', border: '1px solid', borderColor: 'divider', bgcolor: 'background.paper', display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Box sx={{ width: 48, height: 48, borderRadius: '12px', bgcolor: s.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                          <Typography variant="h5" fontWeight={800} sx={{ color: s.color }}>{typeof s.value === 'number' ? s.value : s.value.split('%')[0]}<Typography component="span" variant="body2" sx={{ color: s.color }}>%</Typography></Typography>
                      </Box>
                      <Box>
                          <Typography variant="h5" fontWeight={800}>{s.value}{s.unit}</Typography>
                          <Typography variant="body2" color="text.secondary">{s.label}</Typography>
                      </Box>
                  </Paper>
              </Grid>
          ))}
      </Grid>

      {/* Forms Layout */}

      <Box component="form" noValidate autoComplete="off">
          
        {/* Badges Section */}
        <Box sx={{ mb: 6 }}>
             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3,  borderBottom: '1px solid', borderColor: 'divider', pb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <EmojiEventsIcon sx={{ color: '#f59e0b' }} />
                    <Typography variant="h5" fontWeight={800} color="text.primary">Badges & Achievements</Typography>
                </Box>
                <Button size="small" sx={{ fontWeight: 700, textTransform: 'none', color: 'primary.main', borderRadius: '8px' }}>View All</Button>
             </Box>
             <Grid container spacing={3}>
                {[
                    { icon: <MilitaryTechIcon sx={{ fontSize: 32 }} />, color: 'warning', title: 'Math Whiz', sub: 'Top 10% in Calculus' },
                    { icon: <PsychologyIcon sx={{ fontSize: 32 }} />, color: 'secondary', title: 'Consistent Mind', sub: `${stats.streak_days} Day Streak` },
                    { icon: <RocketLaunchIcon sx={{ fontSize: 32 }} />, color: 'info', title: 'Fast Learner', sub: 'Completed 5 modules' },
                ].map((badge, idx) => (
                    <Grid item xs={12} sm={6} md={3} key={idx}>
                         <Paper elevation={0} sx={{ p: '24px !important', borderRadius: '16px', border: '1px solid', borderColor: 'divider', bgcolor: 'background.paper', textAlign: 'center', transition: 'all 0.2s', '&:hover': { transform: 'translateY(-4px)', borderColor: 'primary.main', boxShadow: '0 10px 20px -10px rgba(0,0,0,0.5)' } }}>
                             <Avatar sx={{ width: 56, height: 56, bgcolor: `${badge.color}.light`, color: `${badge.color}.main`, mx: 'auto', mb: 2, borderRadius: '12px' }}>
                                 {badge.icon}
                             </Avatar>
                             <Typography variant="subtitle1" fontWeight={800} color="text.primary">{badge.title}</Typography>
                             <Typography variant="body2" color="text.secondary" fontWeight={500}>{badge.sub}</Typography>
                         </Paper>
                    </Grid>
                ))}
                 <Grid item xs={12} sm={6} md={3}>
                    <Paper elevation={0} variant="outlined" sx={{ p: '24px !important', borderRadius: '16px', border: '1px dashed', borderColor: 'divider', textAlign: 'center', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', bgcolor: 'transparent' }}>
                         <Avatar sx={{ width: 56, height: 56, bgcolor: 'rgba(255,255,255,0.05)', color: 'text.disabled', mb: 2, borderRadius: '12px' }}>
                             <LockIcon fontSize="large" />
                         </Avatar>
                         <Typography variant="body1" fontWeight={700} color="text.secondary">Next Reward</Typography>
                         <Typography variant="body2" color="text.disabled" fontWeight={500}>Level 6 Scholar</Typography>
                    </Paper>
                </Grid>
             </Grid>
        </Box>

        {/* Personal Info */}
        <Box sx={{ mb: 6 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4, borderBottom: '1px solid', borderColor: 'divider', pb: 2 }}>
                <BadgeIcon color="primary" />
                <Typography variant="h5" fontWeight={800} color="text.primary">Personal Information</Typography>
            </Box>
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <TextField label="First Name" fullWidth defaultValue={user?.first_name || "Alex"} variant="outlined" sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }} />
                </Grid>
                <Grid item xs={12} md={6}>
                    <TextField label="Last Name" fullWidth defaultValue={user?.last_name || "Johnson"} variant="outlined" sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }} />
                </Grid>
                <Grid item xs={12} md={12}>
                    <TextField label="Email Address" fullWidth defaultValue={user?.email || "alex@example.com"} variant="outlined" sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }} />
                </Grid>
                <Grid item xs={12}>
                    <TextField 
                        label="Bio / About Me" 
                        fullWidth 
                        multiline 
                        rows={3} 
                        value={profile.bio} 
                        onChange={(e) => handleChange('bio', e.target.value)}
                        helperText="250 characters left"
                        variant="outlined"
                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '16px' } }}
                    />
                </Grid>
            </Grid>
        </Box>

        {/* Academic Details */}
        <Box sx={{ mb: 6 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4, borderBottom: '1px solid', borderColor: 'divider', pb: 2 }}>
                <SchoolIcon color="primary" />
                <Typography variant="h5" fontWeight={800} color="text.primary">Academic Details</Typography>
            </Box>
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <TextField 
                        label="School / University" 
                        fullWidth 
                        value={profile.school} 
                        onChange={(e) => handleChange('school', e.target.value)}
                        variant="outlined"
                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
                    />
                </Grid>
                <Grid item xs={12} md={6}>
                    <TextField 
                        select 
                        label="Current Grade/Level" 
                        fullWidth 
                        value={profile.grade}
                        onChange={(e) => handleChange('grade', e.target.value)}
                        variant="outlined"
                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
                    >
                        <MenuItem value="9th Grade">9th Grade</MenuItem>
                        <MenuItem value="10th Grade">10th Grade</MenuItem>
                        <MenuItem value="11th Grade">11th Grade</MenuItem>
                        <MenuItem value="12th Grade">12th Grade</MenuItem>
                        <MenuItem value="Undergraduate">Undergraduate</MenuItem>
                    </TextField>
                </Grid>
            </Grid>
        </Box>

        {/* AI Preferences */}
        <Box sx={{ mb: 6 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4, borderBottom: '1px solid', borderColor: 'divider', pb: 2 }}>
                <AutoAwesomeIcon color="primary" />
                <Typography variant="h5" fontWeight={800} color="text.primary">AI Learning Preferences</Typography>
            </Box>
            <Paper elevation={0} variant="outlined" sx={{ p: '32px !important', borderRadius: '16px', borderColor: 'divider', bgcolor: 'background.paper' }}>
                <Grid container spacing={4}>
                    <Grid item xs={12} md={8}>
                        <Typography variant="subtitle1" fontWeight={700} color="text.primary">Target Subjects</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 3, mt: 0.5 }}>Select subjects for AI daily recommendations.</Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            {profile.subjects.map(sub => (
                                <Chip key={sub} label={sub} onDelete={() => {}} sx={{ fontWeight: 600, borderRadius: '8px', bgcolor: 'rgba(19, 127, 236, 0.1)', color: 'primary.main', border: '1px solid', borderColor: 'primary.main' }} />
                            ))}
                            <Chip icon={<AddIcon />} label="Add Subject" onClick={() => {}} variant="outlined" sx={{ fontWeight: 600, borderRadius: '8px', color: 'text.primary', borderColor: 'divider', '&:hover': { bgcolor: 'rgba(255,255,255,0.05)' } }} />
                        </Box>
                    </Grid>
                    <Grid item xs={12}><Divider /></Grid>
                    {[
                        { label: 'Adaptive Difficulty', value: 'adaptiveDifficulty', desc: 'AI adjusts question hardness based on performance.' },
                        { label: 'Strict Exam Mode', value: 'strictMode', desc: 'Disable hints and timer pauses during practice exams.' }
                    ].map((pref) => (
                        <Grid item xs={12} key={pref.value}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="subtitle1" fontWeight={700} color="text.primary">{pref.label}</Typography>
                                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>{pref.desc}</Typography>
                                </Box>
                                <Switch checked={profile.preferences[pref.value]} onChange={() => handlePreferenceChange(pref.value)} color="primary" />
                            </Box>
                        </Grid>
                    ))}
                </Grid>
            </Paper>
        </Box>

        {/* Notifications */}
        <Box sx={{ mb: 6 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4, borderBottom: '1px solid', borderColor: 'divider', pb: 2 }}>
                <NotificationsActiveIcon color="primary" />
                <Typography variant="h5" fontWeight={800} color="text.primary">Notification Preferences</Typography>
            </Box>
             <Paper elevation={0} variant="outlined" sx={{ borderRadius: '16px', overflow: 'hidden', borderColor: 'divider', bgcolor: 'background.paper' }}>
                {[
                    { icon: <ScheduleIcon color="primary" />, label: 'Study Reminders', desc: 'Daily nudges to complete your study goals', key: 'studyReminders' },
                    { icon: <EmojiEventsIcon sx={{ color: '#f59e0b' }} />, label: 'Achievement Unlocks', desc: 'Get notified when you earn new badges', key: 'achievementUnlocks' },
                    { icon: <MailIcon sx={{ color: '#0bda5b' }} />, label: 'Weekly Progress Report', desc: 'Receive a summary via email every Monday', key: 'weeklyReport' },
                ].map((item, i) => (
                    <Box key={item.key} sx={{ p: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: i < 2 ? '1px solid' : 'none', borderColor: 'divider', transition: 'background-color 0.2s', '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' } }}>
                        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                            <Box sx={{ width: 40, height: 40, borderRadius: '10px', bgcolor: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                {item.icon}
                            </Box>
                            <Box>
                                <Typography variant="subtitle2" fontWeight={700} color="text.primary">{item.label}</Typography>
                                <Typography variant="body2" color="text.secondary" fontWeight={500}>{item.desc}</Typography>
                            </Box>
                        </Box>
                        <Switch checked={profile.preferences[item.key]} onChange={() => handlePreferenceChange(item.key)} color="primary" />
                    </Box>
                ))}
            </Paper>
        </Box>

        {/* Linked Accounts */}
        <Box sx={{ mb: 6 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4, borderBottom: '1px solid', borderColor: 'divider', pb: 2 }}>
                <LinkIcon color="primary" />
                <Typography variant="h5" fontWeight={800} color="text.primary">Linked Accounts</Typography>
            </Box>
             <Paper elevation={0} variant="outlined" sx={{ borderRadius: '16px', overflow: 'hidden', borderColor: 'divider', bgcolor: 'background.paper' }}>
                 {/* Google */}
                 <Box sx={{ p: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid', borderColor: 'divider' }}>
                     <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                         <Avatar sx={{ bgcolor: 'white', p: 0.5, width: 40, height: 40, borderRadius: '10px' }}>
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuC5uqCYrxY_4EjKRXpTwNc8pnrdTqWkzqdUGGp7JZku9a8BscAGdaVSpmZBGfwsvNnmUh6x4hTJuTuIdp7VOGhZxTN3LL-WatZMjQuPri6Gf4Gylt2C2SYV9dAWmcPxWByb5CM5l9zBXbWuXQdKs0988UMZ2oUELviMlUi1gP3KZIhiVHyNJBhrTtP-rOq6svyi5gnqoGr6rRbxsfZ3D_HfQStF0ll7N9UnzslLCcHmSObPPzMBFDXG7GRTPHrmVJAaSEX33Et98cI" width="24" height="24" alt="G" />
                         </Avatar>
                         <Box>
                             <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="subtitle1" fontWeight={700} color="text.primary">Google</Typography>
                                <Chip label="Connected" size="small" color="success" sx={{ height: 24, fontSize: '0.75rem', fontWeight: 700, borderRadius: '6px', bgcolor: 'rgba(11, 218, 91, 0.1)', color: '#0bda5b' }} />
                             </Box>
                             <Typography variant="body2" color="text.secondary" fontWeight={500}>{user?.email}</Typography>
                         </Box>
                     </Box>
                     <Button sx={{ color: '#ef4444', fontWeight: 700, textTransform: 'none', borderRadius: '8px' }}>Disconnect</Button>
                 </Box>
                 {/* GitHub */}
                  <Box sx={{ p: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                     <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                         <Avatar sx={{ bgcolor: 'white', color: 'black', width: 40, height: 40, borderRadius: '10px' }}>
                            <GitHubIcon />
                         </Avatar>
                         <Box>
                             <Typography variant="subtitle1" fontWeight={700} color="text.primary">GitHub</Typography>
                             <Typography variant="body2" color="text.secondary" fontWeight={500}>Link repositories for coding assignments</Typography>
                         </Box>
                     </Box>
                     <Button sx={{ color: 'text.primary', fontWeight: 700, textTransform: 'none', borderRadius: '8px', border: '1px solid', borderColor: 'divider' }}>Connect</Button>
                 </Box>
             </Paper>
        </Box>

        {/* Footer Actions */}
        <Box sx={{ 
            position: 'fixed', bottom: 0, left: 0, right: 0, 
            bgcolor: 'background.paper', borderTop: '1px solid', borderColor: 'divider', 
            p: 2, zIndex: 10, display: 'flex', justifyContent: 'flex-end', gap: 2 
        }}>
            <Container maxWidth="lg" sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                <Button variant="text" size="large" sx={{ fontWeight: 700, textTransform: 'none', borderRadius: '12px', color: 'text.secondary' }}>Cancel</Button>
                <Button 
                    variant="contained" 
                    size="large" 
                    sx={{ px: 4, fontWeight: 700, borderRadius: '12px', textTransform: 'none', boxShadow: 'none' }}
                    onClick={handleSave}
                    disabled={saving}
                >
                    {saving ? 'Saving...' : 'Save Changes'}
                </Button>
            </Container>
        </Box>

      </Box>
    </Box>
  );
}
